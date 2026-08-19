# radar2 — Ouster ↔ radar2, two rounds (2026-08-19)

**Not deployable, and not for want of poses.** 42 captures over two rounds; the
geometry is degenerate and more of the same will not fix it. What it needs is a
tape measure. Poses: `2026-08-19_ouster_radar2_poses.json`.

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

## The two rounds disagree — and the split-half test says why

| | t (cm) | \|t\| | inliers | residual | LOO | cond |
|---|---|---|---|---|---|---|
| round 1 (21) | [+5.9, −9.0, +8.2] | 13.5 | 13/21 | 1.40σ | 2.06σ | 10.0 |
| round 2 (21) | [+1.5, **−34.3**, +6.1] | 34.9 | 16/21 | **1.17σ** | **1.35σ** | 8.4 |
| merged (42) | [+3.9, −16.8, +8.9] | 19.4 | 27/42 | 1.36σ | 1.50σ | 8.6 |

Round 2 is internally the healthiest solve in this whole project — residual 1.17σ,
LOO 1.35σ. It is also **25.7 cm and 38.9° away from round 1**. Both cannot be right.

The discriminator is to split a *single* round in half at random and see how far
the halves land from each other. If a within-round split scatters as much as the
two rounds do, the geometry is degenerate; if it does not, the rig moved.

| set | half-vs-half t gap | rot gap |
|---|---|---|
| radar1, 54 poses | **8.4 cm** | **9.8°** |
| radar1, first 21 poses | 23.2 cm | 28.0° |
| radar2 round 1 (21) | 42.2 cm | 107.4° |
| radar2 round 2 (21) | 33.0 cm | 37.2° |
| radar2, both rounds (42) | 26.0 cm | 22.5° |
| *round 1 vs round 2, for reference* | *25.7 cm* | *38.9°* |

Splitting one round scatters **further than the two rounds differ**. The rig did
not move. The problem is that any subset of these captures gives a different
answer.

## Why: the roll turns the sensor's narrow axis sideways

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

Simulating a radar2 with a genuinely dead elevation channel, varying only how far
the targets spread horizontally:

| horizontal spread | half-vs-half t gap | rot gap |
|---|---|---|
| 15° (what the mount allows) | 9.3 cm | 5.2° |
| 50° | 6.8 cm | 8.7° |
| 110° | **4.8 cm** | **4.1°** |

It wants roughly 50°. The mount permits 40° total. **Structurally short — this is
not a collection-technique problem.**

## The fix is a tape measure, not more poses

This is the same situation `CALIBRATION_PROTOCOL.md` §0b describes for a single-
reflector radar's blind axis: information the sensor cannot supply has to come
from outside. Feeding the offline solver a prior of the stated width, and
re-running the split-half test on the same 42 poses:

| prior | half-vs-half t gap | rot gap |
|---|---|---|
| none | 26.7 cm | 24.0° |
| position ±5 cm | 8.1 cm | 16.7° |
| position ±3 cm | **3.9 cm** | 19.3° |
| position ±3 cm + rotation ±10° | 4.6 cm | **10.8°** |

A ±3 cm tape measure takes the scatter from 26.7 cm to 3.9 cm — better than
radar1 managed with 54 poses and no prior.

```bash
ros2 run wicoms_utils radar_lidar_calib --ros-args \
  -p radar_topic:=/radar2/radar/points_all -p pc_field_snr:=intensity \
  -p radar_name:=radar2 -p child_frame:=radar2_link \
  -p reflector_size:=0.19 -p debug_scale:=0.7 \
  -p use_extrinsic_prior:=true \
  -p prior_t_xyz:="[<x>,<y>,<z>]"     `# radar2 position in the LIDAR frame, tape` \
  -p prior_t_sigma_m:=0.03 \
  -p prior_rpy_deg:="[<r>,<p>,<y>]"   `# nominal mounting, rough is fine` \
  -p prior_rot_sigma_deg:=10.0
```

The existing 42 poses can be re-solved with the prior offline once the tape number
exists — no recollection needed.

## A correction to the round-1 advice

After round 1 the note here said to stay inside ±10° horizontally and not chase
the `el` bars, on the grounds that the far-off-axis captures were being rejected.
That reasoning was wrong: horizontal spread is precisely what breaks the planar
degeneracy, and those captures were being rejected because the dead channel
mispredicts them, not because they carried bad information. It changed nothing in
practice — the mount only reaches ±20° either way — but the principle matters if
this radar is ever re-mounted.

## Numbers for comparison later

```
13/21 inliers   residual 1.40s   LOO 2.06s   cond 10.0
t (m)  : +0.0589 -0.0903 +0.0819   |t| 13.5 cm
quat   : -0.0282 +0.6206 +0.7798 +0.0769
1s rot : 5.53 6.70 5.96 deg     1s t : 15.6 74.5 72.9 mm
```

Only the range direction (15.6 mm) is trustworthy. Judge radar2 on its vertical
and range once round 2 is in; its horizontal will stay soft by construction, and
radar1 covers it.
