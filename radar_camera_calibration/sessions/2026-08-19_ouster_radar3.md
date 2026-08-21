# radar3 — Ouster ↔ radar3, Arducam (2026-08-19)

33 poses over two runs (12 + 21). **Data only — no priors of any kind.**
Poses: `2026-08-19_ouster_radar3_poses.json`.

```bash
# os_lidar -> radar3_link
ros2 run tf2_ros static_transform_publisher \
  0.042951 -0.107746 0.160631  0.002367 0.983971 0.000289 0.178315 \
  os_lidar radar3_link

# composed: arducam_optical_frame -> radar3_link
ros2 run tf2_ros static_transform_publisher \
  0.011628 -0.207269 -0.156893  0.577560 0.568043 0.417338 -0.411802 \
  arducam_optical_frame radar3_link
```

```
r3_t_xyz:="[0.0116,-0.2073,-0.1569]"  r3_quat_xyzw:="[0.5776,0.5680,0.4173,-0.4118]"
```

| | value |
|---|---|
| t (m), lidar frame | [+0.042951, −0.107746, +0.160631] · \|t\| 19.8 cm |
| quat xyzw | [+0.002367, +0.983971, +0.000289, +0.178315] |
| 1σ rot | 1.54 / 2.31 / 3.93 deg |
| 1σ t | 14.4 / 26.2 / 49.2 mm |
| residual / LOO | 1.27σ / 1.36σ |
| condition number | 6.1 |
| inliers | 28/33 (85%) |
| split-half t gap | 15.0 cm |

Mount: **inverted** — its +Z is 155° from world up, boresight pitched ~20° down.
Behaviourally that makes it a radar1, not a radar2: good azimuth axis horizontal,
dead elevation axis **vertical**.

```
X fwd  (range)          -> [-0.94  0.00 -0.35]
Y left (AZIMUTH, good)  -> [ 0.00 +1.00  0.00]   horizontal
Z up   (ELEVATION, dead)-> [+0.35  0.00 -0.94]   vertical

channels: az slope +0.91    el slope +0.10
```

## An unresolved disagreement with the tape

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

At 33 poses a chi-square test cannot rule the tape out (Δχ² ≈ 24, suggestive).
At ~70 poses it becomes decisive (Δχ² ≈ 40). That is the way to settle it.

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

radar1 and radar2 are final. radar3 is usable and has the best residual and inlier
rate of the three, but the weakest split-half and an unexplained 14 cm depth
disagreement. ~35 more poses at a wider azimuth (69° → ~96°) would roughly halve
its uncertainties and settle the depth question.
