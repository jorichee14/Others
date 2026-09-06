# Stage 08 ablation - mobile_1 ZED (`arms` track)

Three trajectories from ONE estimator over the same nodes; the arms differ only
in which residual blocks are active.

| arm | odometry | map ICP | board factors |
|---|---|---|---|
| `A_icp`    | yes | yes | no  |
| `B_boards` | yes | no  | yes |
| `C_joint`  | yes | yes | yes |

Only the **off-diagonal** cells are held out: `A_icp` board resid and
`B_boards` map rms. Everything else is training error.

## Run 1 - 92 board factors, C started from the odometry chain

```
arm              board resid (cm)        map rms (cm)          vs C (cm)
A_icp             28.6 med   68.5 p95    6.66 med 11.54 p95     0.0 med   34.8 max
B_boards           0.1 med    0.3 p95   15.72 med 19.17 p95   262.7 med  488.0 max
C_joint            5.1 med   64.6 p95    6.35 med 11.54 p95     0.0 med    0.0 max
```

## Run 2 - 372 board factors (resolve_instances fixed), C still chain-started

```
arm              board resid (cm)        map rms (cm)          vs C (cm)
A_icp           1113.0 med 1154.0 p95    6.23 med 12.75 p95    25.0 med 1155.0 max
B_boards           0.0 med    0.9 p95    7.17 med 13.76 p95   171.3 med  766.3 max
C_joint            3.3 med   47.4 p95    5.40 med 12.95 p95     0.0 med    0.0 max
```

## Run 3 - C initialised from arm B (current code)

1178 nodes, 372 board factors, 864 clouds.

```
arm              board resid (cm)        map rms (cm)          vs C (cm)
A_icp           1113.0 med 1154.0 p95    6.23 med 12.75 p95   648.8 med 1197.6 max
B_boards           0.0 med    0.9 p95    7.17 med 13.76 p95    13.1 med   97.9 max
C_joint            2.5 med    5.9 p95    5.88 med 11.41 p95     0.0 med    0.0 max

arm A_icp      vs lidar: median 595.7 cm  p95 815.3 cm
arm B_boards   vs lidar: median 466.9 cm  p95 882.9 cm
arm C_joint    vs lidar: median 431.0 cm  p95 883.0 cm
```

## What the boards buy

1. **A gross error ICP cannot see.** Arm A: held-out board residual 1113 cm
   median at 6.23 cm map rms. Geometry alone sits 11 m from the fiducial
   position while reporting a 6 cm fit - a wrong-basin lock, on real data.
2. **Agreement with unseen geometry.** Arm B: held-out map rms 7.17 cm vs
   6.23 cm for the arm fit to the map. Boards recover map-consistent shape
   they were never given.
3. **More board evidence improves the held-out cell.** Run 1 -> 2 quadrupled
   the board factors (92 -> 372); arm B map rms 15.72 -> 7.17 cm (-54%).
4. **ICP still adds on top.** C vs B: map rms 7.17 -> 5.88 cm (-18%) for a
   board resid cost of 0.0 -> 2.5 cm. The constraints trade off.
5. **Initialisation is the dominant effect.** Run 2 -> 3 changed only C's
   start pose: board resid p95 47.4 -> 5.9 cm, B<->C agreement 171.3 ->
   13.1 cm. Boards choose the basin; ICP refines inside it but cannot find it.

## Caveats

- A's board resid is NOT comparable between runs 1 and 2: run 1 evaluated
  against 92 sightings near the session start, run 2 against 372 spanning the
  whole session. The jump is metric coverage, not a regression.
- The `vs lidar` rows compare against the LiDAR track BEFORE `seed_from` was
  added, i.e. against the wrong-basin LiDAR trajectory. They are a record of
  the disagreement that motivated `seed_mode`, not an accuracy result.
- mobile_2 has never been run: its track is skipped for a missing
  `cam_extrinsic_xyzquat` (`ros2 run tf2_ros tf2_echo camera_link
  camera_color_optical_frame`).
- The LiDAR track (`lidar_icp`) has no arms - it is geometry-only by
  construction.
