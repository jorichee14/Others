# radar3 — Ouster ↔ radar3, Arducam (2026-08-19)

33 poses over two runs (12 + 21), solved with a **tape prior on the vertical only**.
Poses: `2026-08-19_ouster_radar3_poses.json`.

```bash
# os_lidar -> radar3_link
ros2 run tf2_ros static_transform_publisher \
  0.0577 -0.1097 0.0717  0.00310 0.98794 0.00233 0.15480 \
  os_lidar radar3_link

# composed: arducam_optical_frame -> radar3_link
ros2 run tf2_ros static_transform_publisher \
  0.011012 -0.117821 -0.168001  0.568811 0.556671 0.431578 -0.424632 \
  arducam_optical_frame radar3_link
```

```
r3_t_xyz:="[0.0110,-0.1178,-0.1680]"  r3_quat_xyzw:="[0.5688,0.5567,0.4316,-0.4246]"
```

| | value |
|---|---|
| t (m), lidar frame | [+0.0577, −0.1097, +0.0717] · \|t\| 14.3 cm |
| quat xyzw | [+0.00310, +0.98794, +0.00233, +0.15480] |
| 1σ rot | 1.64 / 1.83 / 3.98 deg |
| 1σ t | 11.9 / 28.1 / 19.0 mm |
| residual / LOO | 1.29σ / 1.36σ |
| condition number | 6.1 |
| inliers | 28/33 |
| split-half t gap | 9.6 cm |

Mount: **inverted** — its +Z is 155° from world up, boresight pitched ~20° down.
Behaviourally that makes it a radar1, not a radar2: good azimuth axis horizontal,
dead elevation axis **vertical**.

```
X fwd  (range)          -> [-0.94  0.00 -0.35]
Y left (AZIMUTH, good)  -> [ 0.00 +1.00  0.00]   horizontal
Z up   (ELEVATION, dead)-> [+0.35  0.00 -0.94]   vertical

channels: az slope +0.91    el slope +0.10
```

## Why the vertical came from a tape

The free solve put radar3 **21.3 cm above the Arducam**; a tape said **11–12 cm**.
That is 2.0σ on the one axis this radar cannot measure, so the disagreement was
expected to land exactly there.

The Arducam transform was cleared first, not assumed. It is 7.4 cm from the ZED
left lens by the published transforms, and that was confirmed by tape. A 10 cm
error in the Arducam's Z would have made that distance ~15.2 cm, so the Arducam is
sound and the error is radar3's `t_z`.

Applying an 11.5 cm ±2 cm prior on the vertical alone:

| | free | with the tape |
|---|---|---|
| residual | 1.27σ | 1.29σ |
| LOO | 1.36σ | 1.36σ |
| inliers | 28/33 | 28/33 |
| **1σ t_z** | **49.2 mm** | **19.0 mm** |

Moving `t_z` by 9.4 cm cost **0.02σ of residual and nothing in LOO**. A wrong prior
would have blown the fit up; this one did not, which says the data never had an
opinion about that axis and the tape is pure added information. `t_x` and `t_y` were
left free.

It settles at 12.4 cm rather than 11.5 because the prior is soft; tighten it if the
tape can be taken more precisely.

## Still owed

`radar1` has the identical geometry — dead axis vertical, `t_z` 1σ 40.9 mm, no tape
applied. It predicts **16.9 cm below the lidar**. `radar2` is the control: its good
axis *is* vertical, so its predicted **13.9 cm below the lidar** should hold with no
prior at all. If radar1 is ~10 cm out and radar2 is not, the pattern is confirmed
and radar1 gets the same one-line correction.
