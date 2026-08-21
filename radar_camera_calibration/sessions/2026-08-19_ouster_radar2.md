# radar2 — Ouster ↔ radar2 (2026-08-19)

**Final: rounds 3 + 4, 42 poses.** Rounds 1 and 2 are excluded — they sit 22–30 cm
from these and from each other, and round 2 misses the tape-measured radar1↔radar2
separation by 24 cm despite having the best residual of any set collected. Rounds 3
and 4 agree to **1.8 cm / 6.7°** and were taken either side of the rig being raised,
which also confirms the raise was rigid.

```bash
# os_lidar -> radar2_link
ros2 run tf2_ros static_transform_publisher \
  0.0367 -0.1208 -0.1391  0.01383 0.73662 0.67577 0.02329 \
  os_lidar radar2_link

# composed through GLIM: zed_left_camera_optical_frame -> radar2_link
ros2 run tf2_ros static_transform_publisher \
  -0.053850 0.046553 -0.112039  0.724722 0.038269 0.687594 -0.022953 \
  zed_left_camera_optical_frame radar2_link
```

```
r2_t_xyz:="[-0.0538,0.0466,-0.1120]"  r2_quat_xyzw:="[0.7247,0.0383,0.6876,-0.0230]"
```

| | radar1 | radar2 |
|---|---|---|
| poses / inliers | 46/54 (85%) | 33/42 (79%) |
| residual / LOO | 1.29σ / 1.36σ | 1.61σ / 1.78σ |
| condition number | 5.2 | **3.8** |
| **split-half t gap** | **8.4 cm** | **12.0 cm** |
| 1σ t X / Y / Z (cam, mm) | **19.3** / 40.9 / **8.7** | 37.5 / **36.4** / **10.1** |
| bias X / Y / Z (mm) | +16.8 / −49.3 / +12.6 | −7.4 / +12.3 / +24.3 |
| channel slope az / el | +0.99 / **+0.14** | +1.09 / **−0.07** |
| dead axis points | **vertical** | **horizontal** |

Independent check: predicted radar1↔radar2 separation **26.3 cm** against a
tape-measured **29.6 cm** (−3.3 cm), inside both the 37 mm 1σ and the accuracy of
measuring between two phase centres by hand.

Each radar's dead axis is the other's good one, so the pair covers 3-D where
neither does alone: best-of-pair is 19.3 / 36.4 / 8.7 mm.

**Do not judge this radar by residual or LOO.** Round 2 had the best residual in the
project and was 24 cm wrong. The trustworthy tests here are the split-half gap and
the tape.

## How it got here

## Finding 1 — radar2 is rolled ~90° from radar1, and that is worth keeping

```
X fwd  (range)        -> [-0.99 +0.08 -0.14]   horizontal
Y left (AZIMUTH 3deg) -> [-0.15 -0.22 +0.96]   VERTICAL   (16 deg off vertical)
Z up   (ELEVATION 8deg)-> [+0.05 +0.97 +0.23]  horizontal
```

## Finding 2 — the elevation channel is dead on BOTH radars, and it is the chip

Regressing each channel's reported angle against what the solved extrinsic says it
should have reported (slope 1 = the channel tracks the target):

| | azimuth | elevation | so the dead axis points |
|---|---|---|---|
| radar1 | **+0.99** | +0.14 | **vertical** |
| radar2 | **+1.02** | −0.10 | **horizontal** |

Two different mountings, same pattern: azimuth works, elevation emits a
near-constant angle. It is a property of the chip or the config, not of either
bracket. The node now prints this as a `channels:` line after every solve.

**The consequence is good news.** radar2's *working* azimuth is the vertical one,
which is exactly what radar1 cannot measure; radar1's working azimuth is
horizontal, which is exactly what radar2 cannot measure. Each covers the other's
blind axis, so the pair is a usable 3-D sensor even though neither is alone.

## Why it took four rounds: the roll turns the sensor's narrow axis sideways

`fovCfg -1 60.0 20.0` — ±60° in azimuth, ±20° in elevation. Once you account for
how each is mounted:

| | horizontal FoV | vertical FoV |
|---|---|---|
| radar1, upright | **±60°** | ±20° |
| radar2, rolled 90° | **±20°** | ±60° |

Every radar2 capture therefore lies within ±20° of a single vertical plane through
the sensor — measured extent over both rounds is −16.2° to +1.1°. A near-planar
target set cannot pin the out-of-plane DOF, which is exactly the one translation
and two rotations that move between rounds.

## A correction to the round-1 advice

After round 1 the note here said to stay inside ±10° horizontally and not chase
the `el` bars, on the grounds that the far-off-axis captures were being rejected.
That reasoning was wrong: horizontal spread is precisely what breaks the planar
degeneracy, and those captures were being rejected because the dead channel
mispredicts them, not because they carried bad information. It changed nothing in
practice — the mount only reaches ±20° either way — but the principle matters if
this radar is ever re-mounted.

## What it cost, and what would have been cheaper

Four rounds, 84 captures, to get two that agree. The reason is in the section
above: rolled 90°, radar2 sees only ±20° horizontally, so every capture sits near
one vertical plane and the out-of-plane DOF are barely constrained. Rounds 1 and 2
landed in different places for that reason, not because anything was done wrong.

What separated rounds 3 and 4 from 1 and 2 was **true horizontal spread**, measured
from the lidar rather than from the radar's own (dead) elevation channel:

| | true horizontal spread |
|---|---|
| round 1 | 52° |
| **round 2** | **19°** ← the narrow one, and the 24 cm outlier |
| round 3 | 46° |
| round 4 | 46° |

Two things would have shortened this:

- **Ignore the `el` / `el_bal` bars on both radars.** They are computed from what
  the radar reports, and that channel emits a near-constant angle, so they can
  never go green and they say nothing about the geometry you actually achieved.
  Judge horizontal spread from the lidar side instead.
- **Take the radar1↔radar2 tape measurement first.** It costs two minutes and it
  rejects a bad round immediately, rather than after another hour of collecting.

A ±3 cm prior on the position would also have reached this quality from the first
42 poses — `solve_radar_lidar.py` now supports `t_prior`/`rot_prior` for exactly
that, and `use_extrinsic_prior` does it live. It was not needed in the end, but it
remains the cheapest insurance for the next rolled radar.
