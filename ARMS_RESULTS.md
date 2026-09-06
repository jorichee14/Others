# Stage 08 ablation - final run (both agents)

Three trajectories from ONE estimator over the same nodes; the arms differ only
in which residual blocks are active.

| arm | odometry | map ICP | board factors | notes |
|---|---|---|---|---|
| `A_icp`    | yes | yes | no  | geometry only |
| `B_boards` | yes | no  | yes | fiducial only |
| `B_breaks` | yes | no  | yes | + tracking-break edges freed (mobile_1 only) |
| `C_joint`  | yes | yes | yes | |

Only the **off-diagonal** cells are held out: `A_icp` board resid and
`B_*` map rms. Everything else is training error.

## mobile_1 (ZED + Ouster) - VALIDATED

```
arm          board resid (cm)     map rms (cm)      vs C (cm)      vs odom (cm)
A_icp         1.1 med  1.9 p95   4.39 med  5.48    0.0 /   1.0    594.5 / 1074.9
B_boards      0.0 med  1.0 p95  10.67 med 14.79    4.1 / 413.8    667.1 / 1074.4
B_breaks      0.0 med  0.4 p95   9.35 med 13.38    3.4 / 495.0    360.4 / 1074.6
C_joint       1.1 med  1.8 p95   4.39 med  5.48    0.0 /   0.0    594.5 / 1074.9
```

LiDAR track: 1516 scans, 0 unregistered, plane rms median 2.34 cm p95 2.53 cm,
rank-deficient 0.1%. Seeded from the session anchor then its own constant-
velocity prediction - zero scans seeded from odometry.

Three independent confirmations:

1. Both boards land on their survey from the LiDAR track: `anchor` 0.02 m,
   `rs_anchor` 0.01 m, scatter 0.00 m. (In earlier runs `rs_anchor` was 9.20 m
   off - the trajectory moved, not the board.)
2. Extrinsic convention settled by data: through `T_lidar_camera` the sightings
   agree to 0.01 m; through `inv(T_lidar_camera)`, 1.39 m.
3. Two independent estimates of one rigid body (LiDAR ICP vs ZED-with-LiDAR-
   clouds) agree to 0.4 cm median / 1.2 cm p95 / 8.5 cm max, flat over 152 s.

Drift removed, from arm B's own log:

```
t = 15.6 .. 115.2 s (99.6 s, 24.2 m of path, 997 nodes)
odometry drift at re-acquisition 1014.4 cm / 91.3 deg
end gap 1055.4 cm over 25.9 m of path = 40.72% of distance travelled
```

`B_breaks` is the better board-only arm: freeing the 33 tracking-break edges
takes the honest cell from 10.67 -> 9.35 cm and halves the applied correction
(667 -> 360 cm), by not smearing a tracking jump over a 24 m stretch.

**Negative result: C is identical to A to two decimals.** For an agent with
LiDAR, 372 board factors change the solution by at most 1 cm.

## mobile_2 (D455) - ONE ROOT-CAUSE DEFECT

```
arm          board resid (cm)     map rms (cm)      vs C (cm)      vs odom (cm)
A_icp        57.9 med 66.0 p95  15.93 med 21.98   48.7 / 62.3     0.3 /  3.6   <- did NOT converge
B_boards      0.7 med  2.5 p95   9.48 med 13.44    9.4 / 25.5    54.4 / 65.8
C_joint       4.9 med  6.7 p95   5.98 med 10.17    0.0 /  0.0    48.7 / 62.3
```

### Root cause: the depth cloud is 15 cm off the map at a pose containing no ICP

```
depth extrinsic configured   plane rms 15.7 cm,  82% of points on map cells
depth extrinsic inverted     plane rms 14.9 cm,  86%
depth extrinsic identity     plane rms 15.3 cm,  84%
```

All three conventions within 0.8 cm - a 59 mm extrinsic cannot make 15 cm, and
swapping it changes nothing. NOT the extrinsic.

### Corroboration: stages 06 and 08 disagree by 25 cm about this camera

```
rs_anchor  n=56  err med 0.25 m (min 0.22 max 0.26)  seen t=0..13 s
```

At t~0 the anchored odometry IS stage 06's session anchor, so 06 and 08 place
the RealSense 25 cm apart from the same camera seeing the same board over
overlapping time. The tight 0.22-0.26 m band is a systematic offset, not
motion. Stage 06's 6.9 mm was PRECISION, not accuracy - the same trap as the
old 9.20 m `rs_anchor` at 0.02 m scatter.

At 0.75 m range a 0.25 m error is a ~33% range scale error, which points at
intrinsics or marker geometry. Check, in order:

1. Does 06 undistort the mobile_2 color images? The 08 track sets
   `"rectified": false`; if 06 assumes rectified the board pose scales.
2. Same `fx`? 08 reads `fx=645.6` from the bag CameraInfo. A calibration file
   at another resolution is a direct range scale.
3. Same marker/square length for `rs_anchor` in both configs.

Diagnostic: print `||t||` of `T_board_cam` for `rs_anchor` from each stage at
the same stamp. A ratio means scale; a constant means a frame.

### Symptom: arm A did not converge

Every LM step was rejected or made the cost worse; it ended at 23379.0 having
started at 19282.9, and `vs odom` is 0.3 cm. Arm A here IS the anchored
odometry with the map factors inert - its 57.9 cm board residual measures the
odometry, not an ICP arm. Report it as non-converged, do not quote it as A.

Follows from the root cause: 853/1434 frames unregistered (59%), 96.6%
rank-deficient, no usable gradient.

### But C_joint is genuinely the right answer for this agent

C beats A on the board cell (57.9 -> 4.9 cm) and beats B on the map cell
(9.48 -> 5.98 cm). It is the only arm acceptable in both held-out cells.

**The joint arm matters exactly where the geometry is weak.** For the LiDAR
agent C is redundant with A; for the narrow-FOV depth agent, geometry alone
fails outright and only the joint solution passes both cells. That is the
cooperative-perception claim, stated more precisely than "both is better".

Caveat: C's 5.98 cm map rms is computed against clouds that are themselves
15 cm off at a known pose. Fixing the root cause will move this number.

## Open

- The mobile_2 depth/anchor 15-25 cm offset (above). Blocks the mobile_2
  reference.
- mobile_1 ZED odometry: with the 33 flagged steps replaced by ICP increments,
  2.52 m of divergence remains over 25.9 m - slow scale drift or a residual
  frame error. Does not affect arms A or C (neither uses odometry for
  position), but it bounds arm B inside the 99.6 s sighting gap, which is
  where its p95 of 413.8 cm comes from.
- Board coverage: mobile_1 has a 99.6 s / 24.2 m stretch with no board in view.
  A placement fix, not an algorithm fix.
