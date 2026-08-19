# radar2 round 1 — 21 poses, kept for merging (2026-08-19)

Not deployable on its own. Recorded here so round 2 can be merged onto it, and so
the two findings below do not have to be rediscovered.

Poses: `2026-08-19_ouster_radar2_poses.json` · re-solve with `solve_radar_lidar.py`.

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

## What round 2 needs — the opposite of what the bars say

The coverage bars are labelled in RADAR coordinates, and radar2 is rolled, so:

- `az` (bar says 44/60) is **vertical** spread — the **working** axis, worth filling.
  Coverage is lopsided: **15 poses above boresight, only 6 below**. Collect LOW.
- `el` (bar says 16/30) is **horizontal** spread — the **dead** axis. Filling it
  made things worse: of 8 rejects, #6, #17 and #19 all sat 14–16° off-axis
  horizontally with 1.7–2.0 m errors, because a dead channel mislocates
  far-off-axis targets badly. **Stay inside about ±10° horizontally.**
- `near` — more captures under 1.5 m, as always.

Target ~35 poses total so the inlier count clears 25.

Do **not** fix this by widening `sigma_el`. Tried: 8 → 25 buys inliers 13 → 16 and
cond 10.0 → 4.4, but LOO degrades 2.06 → 4.4 and t_y shifts 7 cm. It papers over a
thin capture set instead of thickening it.

## Round 1 numbers, for comparison later

```
13/21 inliers   residual 1.40s   LOO 2.06s   cond 10.0
t (m)  : +0.0589 -0.0903 +0.0819   |t| 13.5 cm
quat   : -0.0282 +0.6206 +0.7798 +0.0769
1s rot : 5.53 6.70 5.96 deg     1s t : 15.6 74.5 72.9 mm
```

Only the range direction (15.6 mm) is trustworthy. Judge radar2 on its vertical
and range once round 2 is in; its horizontal will stay soft by construction, and
radar1 covers it.
