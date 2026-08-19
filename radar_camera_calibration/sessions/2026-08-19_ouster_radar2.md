# radar2 — Ouster ↔ radar2 (IWR6843ISK, rolled 90°), 2026-08-19

**Done.** Four rounds collected; the answer is rounds **3 + 4**, 42 poses. Rounds 1
and 2 are excluded — they sit 22–30 cm from these and from each other, and round 2
misses the tape-measured radar1↔radar2 separation by 24 cm.

Poses: `2026-08-19_ouster_radar2_poses.json` · re-solve with `solve_radar_lidar.py`.

## Result

```
T_os_lidar_radar2   t = [+0.0367 -0.1208 -0.1391]   |t| 18.8 cm
                    q = [+0.01383 +0.73662 +0.67577 +0.02329]

T_cam_radar2        t = [-0.053850 +0.046553 -0.112039]   |t| 13.3 cm
                    q = [+0.724722 +0.038269 +0.687594 -0.022953]
```

```
r2_t_xyz:="[-0.0538,0.0466,-0.1120]"  r2_quat_xyzw:="[0.7247,0.0383,0.6876,-0.0230]"
```

| camera axis | 1σ rot | 1σ t | bias | RMS |
|---|---|---|---|---|
| X (right) | 3.43° | 37.5 mm | −7.4 mm | 513 mm |
| Y (down) | 3.87° | 36.4 mm | +12.3 mm | 197 mm |
| Z (forward) | 2.95° | **10.1 mm** | +24.3 mm | 68 mm |

residual 1.61σ · LOO 1.78σ · cond 3.8 · 33/42 inliers

## How the four rounds went, and why only two of them count

| | inliers | resid | LOO | cond | t (cm) | sep vs tape |
|---|---|---|---|---|---|---|
| round 1 | 13/21 | 1.40σ | 2.06σ | 10.0 | [+5.9, −9.0, +8.2] | +4.5 |
| round 2 | 16/21 | **1.17σ** | **1.35σ** | 8.4 | [+1.5, −34.3, +6.1] | **+23.9** |
| round 3 | 18/21 | 1.64σ | 2.15σ | 4.1 | [+3.0, −12.5, −13.7] | −2.8 |
| round 4 | 15/21 | 1.59σ | 2.15σ | 4.7 | [+3.9, −12.7, −15.2] | −2.8 |
| **3 + 4** | 33/42 | 1.61σ | 1.78σ | **3.8** | [+3.7, −12.1, −13.9] | −3.3 |

Pairwise distance between rounds:

| | t gap | rot gap |
|---|---|---|
| r1 vs r2 | 25.7 cm | 38.9° |
| r1 vs r3 | 22.4 cm | 13.9° |
| r2 vs r4 | 30.4 cm | 57.5° |
| **r3 vs r4** | **1.8 cm** | **6.7°** |

Rounds 3 and 4 were collected either side of the whole rig being raised, so their
agreement also confirms the raise was rigid and did not disturb the extrinsic.

**Round 2 is the cautionary one.** It has the best residual and the best
leave-one-out of any set in this project, and it is 24 cm wrong on a quantity that
can be measured with a tape. On this radar, residual and LOO do not detect the
failure mode — the split-half test and the tape do.

## The two tests that actually discriminate

**Split-half** — refit on two random halves and see how far apart they land:

| set | split-half | cond |
|---|---|---|
| round 3 alone | 20.5 cm | 4.1 |
| round 4 alone | 31.1 cm | 4.7 |
| **rounds 3+4** | **9.3 cm** | **3.8** |
| radar1 (54 poses) | 9.3 cm | 5.2 |

radar2 is now as internally stable as radar1, from 42 poses rather than 54.

**The tape** — radar1 and radar2 are 29.6 cm apart, measured. Each candidate
extrinsic predicts that number, and it is the check that convicted round 2:

| radar2 from | predicted separation | error |
|---|---|---|
| round 1 | 34.1 cm | +4.5 |
| round 2 | 53.5 cm | **+23.9** |
| round 3 | 26.8 cm | −2.8 |
| round 4 | 26.8 cm | −2.8 |
| rounds 3+4 | 26.3 cm | −3.3 |

The remaining 3.3 cm sits inside both the 37 mm 1σ and the accuracy of hand-
measuring between two phase centres. Note the tape is a **scalar**: it pins the
distance, not the direction, so it can rule an answer out but cannot confirm one.

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
