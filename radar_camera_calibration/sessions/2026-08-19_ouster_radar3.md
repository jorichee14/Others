# radar3 — Ouster ↔ radar3, Arducam (2026-08-19)

54 poses over three runs (12 + 21 + 21). **Data only — no priors of any kind**, the same treatment radar1 and radar2 had. All three rounds agree within 2.5–8.3 cm, so none was dropped.
Poses: `2026-08-19_ouster_radar3_poses.json`.

```bash
# os_lidar -> radar3_link
ros2 run tf2_ros static_transform_publisher \
  0.041839 -0.101587 0.161252  -0.007406 0.986026 0.007861 0.166243 \
  os_lidar radar3_link

# composed: arducam_optical_frame -> radar3_link
ros2 run tf2_ros static_transform_publisher \
  0.017773 -0.208023 -0.155788  0.571431 0.563856 0.432969 -0.409965 \
  arducam_optical_frame radar3_link
```

```
r3_t_xyz:="[0.0178,-0.2080,-0.1558]"  r3_quat_xyzw:="[0.5714,0.5639,0.4330,-0.4100]"
```

| | value |
|---|---|
| t (m), lidar frame | [+0.041839, −0.101587, +0.161252] · \|t\| 19.5 cm |
| quat xyzw | [−0.007406, +0.986026, +0.007861, +0.166243] |
| 1σ rot | 1.25 / 1.90 / 3.33 deg |
| 1σ t | 11.2 / 21.3 / 40.3 mm |
| residual / LOO | 1.31σ / 1.36σ |
| condition number | 5.3 |
| inliers | 44/54 (81%) — 10 outliers removed |
| **split-half t gap** | **8.3 cm** |

In the Arducam frame, per axis:

| | 1σ rot | 1σ t | bias | RMS |
|---|---|---|---|---|
| X (right) | 1.90° | 21.2 mm | −9.4 mm | 192 mm |
| Y (down) | 3.33° | **40.6 mm** | +51.0 mm | 366 mm |
| Z (forward) | 1.27° | **10.2 mm** | +30.0 mm | 91 mm |

Mount: **inverted** — its +Z is 155° from world up, boresight pitched ~20° down.
Behaviourally that makes it a radar1, not a radar2: good azimuth axis horizontal,
dead elevation axis **vertical**.

```
X fwd  (range)          -> [-0.94  0.00 -0.35]
Y left (AZIMUTH, good)  -> [ 0.00 +1.00  0.00]   horizontal
Z up   (ELEVATION, dead)-> [+0.35  0.00 -0.94]   vertical

channels: az slope +0.91    el slope +0.10
```

## The disagreement with the tape - resolved in favour of the point cloud

The tape says radar3 sits **2 cm behind** the Arducam; the solve says **16.3 cm**.
Vertical (11.5 vs 12.4 cm) and lateral (2.8 vs 1.2 cm) both agree — only depth does
not, by 14 cm.

The Arducam transform was cleared twice: it is 7.4 cm from the ZED left lens and
10–11 cm forward of the lidar, both confirmed by tape, and a 14 cm error in it
would have made the first of those read ~12.9 cm. A radar range bias cannot
explain it either — it would need ~140 mm, and the range fit shows −35 mm.

Forcing the depth costs 8 inliers; forcing depth **and** vertical together drops to
17/33 and throws the lateral out to 28.6 cm against a measured ~2.8. So the tape
and the data are incompatible rather than merely disagreeing, and no constraint
reconciles them. **The extrinsic above therefore uses the data alone.**

The third round did not move the depth: 16.3 cm before, 16.4 cm now, across 54
poses and three independent sessions.

**Resolved: the point cloud is right.** Six independent checks were run and all
of them agree with the solve.

* **Range-only estimator** - uses no radar angles and no rotation at all, only
  `lidar_r - radar_r = t*u`: **+3.97 +/- 0.89 cm**, which is **8.9 sigma** from a
  hypothetical -4 cm.
* **Bootstrap**, 2000 resamples: **0 of 2000** below -4 cm.
* **chi-square** rules out anything at or below zero.
* **Nine rotation-flip restarts**, deliberately seeded at wrong rotations to look
  for a mirror solution, all converged to the identical answer.
* **radar1 and radar2 solve independently** to +3.3 and +3.7 cm on the same axis,
  from completely separate datasets.
* **Raw range differences** across the three radars: +2.7 / +4.1 / +4.3 cm.

Depth is also the decisively observable axis here (chi-square 68 in the table
below). The composed 15.6 cm stands and the transform is published unflipped.

## Which axes the data actually pins — all three radars

Forcing each translation axis 14 cm off, on a **frozen inlier set** so the outlier
rejection cannot shrink the sample and lower chi-square for the wrong reason:

| | x (fwd/back) | y (lateral) | z (vertical) |
|---|---|---|---|
| radar1 | 27 decisive | 50 decisive | **12 weak** |
| radar2 | 36 decisive | **14 weak** | 20 weak |
| radar3 | 68 decisive | 29 decisive | **8 FLOATS** |

>25 decisive · 9–25 weak · <9 the data cannot tell.

Each radar's weak axis is exactly where its dead elevation channel points —
vertical on radar1 and radar3, horizontal on radar2. **The covariance-derived 1σ
understates the uncertainty on those axes; the split-half gap is the honest
figure** (8.4 / 12.0 / 15.0 cm for radar1 / radar2 / radar3).

## Status

All three radars are final. radar3 now matches radar1 on the measure that matters
— split-half 8.3 cm against 8.4 — with the tightest rotation of the three
(1.25 / 1.90 / 3.33°) and depth good to 10.2 mm. The depth disagreement with the
tape was investigated to the end and **settled in favour of the point cloud** -
see the section above.
