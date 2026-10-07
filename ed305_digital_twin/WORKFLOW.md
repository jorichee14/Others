# Camera placement workflow (ED305)

From the cameras as they are now to a calibrated set with no blind spots.
This is phase 1 of [PLAN.md](PLAN.md); later phases (time sync, robots, data)
start once it is done.

```
A. desk          B. room visit (one, with a laptop)          C. desk / room    D. desk
calibration ─┐   ┌─ per flagged camera: ─────────────┐
room size   ─┼─► │ mark ─► record turn ─► put back   │ ─► optimize ─► re-aim ─► re-calibrate ─► verify
coverage    ─┤   │ measure_rotation ─► mounts.yaml   │      ▲                                  │
which cams? ─┘   └───────────────────────────────────┘      └──── not met: relax / more moves ◄┘
```

Rule that orders everything: **turning a camera breaks its extrinsic
calibration.** So all turning (measuring and re-aiming) happens in one visit,
and re-calibration comes after the last turn.

## A. At the desk, before touching anything

| step | do | output |
|---|---|---|
| A1 | Get the calibration from `HCIS-Lab/ED305-cameras` `calibration/`, write it in this repo's format, room frame (README "Putting in the real calibration"). | `configs/cameras.yaml` |
| A2 | Room size and furniture boxes: tape measure, or the robot LiDAR map. | `configs/room.yaml` |
| A3 | `python coverage_map.py --cameras configs/cameras.yaml --room configs/room.yaml` | baseline blind spots |
| A4 | **No blind spots at k?** Stop here: placement is done, go to PLAN phase 2. | |
| A5 | `python optimize_cameras.py --mounts configs/mounts_small.yaml ...` and again with `mounts_wide.yaml`. Every camera either run moves is **flagged**; only flagged cameras need their range measured. | list of flagged cameras |

## B. One room visit

Before: tell people using the cameras that some will move and lose calibration.
Lights on. Laptop on the lab network with this repo.

For each flagged camera (about 10 min each):

| step | do | why |
|---|---|---|
| B1 | `python recording/focus_score.py N` (ED305-cameras repo), note the score. | turning can knock the focus ring |
| B2 | Tape marks on the bracket and a photo. | to put it back exactly |
| B3 | Record its stream (`ffmpeg ... cam_N.mp4`), hold 2 s, turn slowly to each stop: left, right, back; down, up, back. Hold 2 s, stop. | README "Measuring a mount's range" |
| B4 | Back to the marks, tighten. | its calibration stays valid if nothing else moves |
| B5 | `python measure_rotation.py cam_N.mp4 --camera cam_N --cameras configs/cameras.yaml`; "end vs start" under 1 deg, else record again. Paste its `mount.yaml` line into `configs/mounts.yaml`. | measured range |

Then, still in the room:

| step | do | output |
|---|---|---|
| B6 | `configs/mounts.yaml`: measured cameras as measured, every other camera `fixed: true`. | |
| B7 | `python optimize_cameras.py --cameras configs/cameras.yaml --room configs/room.yaml --mounts configs/mounts.yaml --max-moved N` (N = how many you are willing to re-calibrate). | `moves.md`: which camera, turn how far, aim point |
| B8 | With the printed heads (hardware/pan_tilt): set each camera to the "set scales" readings in `moves.md`. Otherwise: tape a marker at its aim point (floor or wall), open its view full screen at `http://192.168.1.188:3000/cameras`, turn until the marker is at the image centre and the image is level, tighten. | cameras re-aimed |
| B9 | `focus_score.py N` again: same as B1, or refocus. | focus kept |

## C. Re-calibrate the moved cameras

Extrinsics only (intrinsics do not change with turning, as long as the focus
ring was not touched, which B9 checks). Use the ED305-cameras calibration
procedure for just these cameras, write them into `configs/cameras.yaml`.

## D. Verify and record

| step | do | done when |
|---|---|---|
| D1 | `coverage_map.py` on the new `cameras.yaml`. | no blind spots at k (or each remaining one listed and accepted) |
| D2 | Triangulate a point no calibration used (a marker on the floor away from the red dot) from the cameras that see it. | within a few cm |
| D3 | Commit `configs/cameras.yaml`, `room.yaml`, `mounts.yaml`, the coverage report. Update the wiki page "Camera location and calibration" (date, which cameras moved). | |

If D1 fails: back to B7 with a larger `--max-moved` or the wide range where
the mount allows. If no move within the mounts fixes it, it is a team decision
(move furniture, move a mount, add a camera), not a re-run.

## Tools still to write

| tool | for | blocked on |
|---|---|---|
| calibration -> `cameras.yaml` converter | A1 | access to `HCIS-Lab/ED305-cameras` (its calibration format) |
| pose check: project the red dot / markers with `cameras.yaml` onto a snapshot of each camera | A1 (is the March calibration still right?) and D2 | nothing |
