# ed305_digital_twin

Digital twin of the ED305 room (16 Basler cameras, see the lab wiki "ED305
Indoor Camera User Guide") for collaborative perception. The plan is
[PLAN.md](PLAN.md); camera placement step by step is [WORKFLOW.md](WORKFLOW.md). v1 so far: coverage / blind spots, placement within each mount's range, and measuring that range from video.

```
ed305/camera.py        pinhole camera in the room frame; loads cameras YAML
ed305/coverage.py      room + target volume, visibility with occlusion, k-coverage,
                       leave-one-out, greedy smallest camera set
ed305/placement.py     aim as heading/tilt, each mount's range of movement, the search
coverage_map.py        command line: report.md, floor.png, slices.png, coverage.npz
ed305/rotation.py      a camera's rotation from its own video (features -> pure rotation)
optimize_cameras.py    command line: re-aim / move cameras within their mounts' ranges
measure_rotation.py    command line: a camera's pan / tilt range from a video of it being turned
hardware/pan_tilt/     3D-printed pan-tilt head with pan / tilt scales (OpenSCAD, STL)
configs/mounts_printed.yaml   the printed head's ranges
configs/room_nominal.yaml     PLACEHOLDER room size, no furniture yet
configs/cameras_nominal.yaml  PLACEHOLDER layout from the wiki map (not a calibration)
configs/mounts_small.yaml  ASSUMED range: +-15 deg re-aim, nothing slides (the default)
configs/mounts_wide.yaml   ASSUMED range: +-45 deg pan, tilt 15-80, ceiling row turns fully
tests/                 pytest
```

```bash
pip install -r requirements.txt
python coverage_map.py                               # nominal layout, k from room yaml (2)
python coverage_map.py --k 3
python coverage_map.py --without cam_4,cam_6         # what if these are off / moved
python coverage_map.py --cameras configs/cameras.yaml --room configs/room.yaml
python optimize_cameras.py                           # best aims within the mount ranges
python optimize_cameras.py --max-moved 3 --k 3
python -m pytest -q tests
```

## Frames

Room frame: origin on the floor at the red dot of the wiki camera map (under the
middle of cams 10 and 11), z up, x toward the right wall, y toward the top (door)
wall of the floor plan. Camera frame: OpenCV (x right, y down, z forward).

## Putting in the real calibration

Write `configs/cameras.yaml` from the ED305-cameras `calibration/` results.
Per camera, intrinsics (`K` 3x3 or `fx`/`fy`/`cx`/`cy`, `width`, `height`,
optional `dist` = OpenCV k1 k2 p1 p2 k3) and one pose:

```yaml
defaults: {width: 1920, height: 1080}
cameras:
  cam_1:
    K: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    dist: [k1, k2, p1, p2, k3]
    T_world_cam: [[...], [...], [...], [0, 0, 0, 1]]   # camera -> room
    # or  rvec: [...]  tvec: [...]                       # OpenCV room -> camera (solvePnP)
  cam_2: ...
```

If the calibration is not in this frame, move it there first (one rigid
transform for all cameras). `enabled: false` leaves a camera out.

Then measure the room (`room.x`, `room.y`, `room.height`) and add the furniture
as `obstacles` boxes in `configs/room.yaml`.

## Optimizing camera placement

`optimize_cameras.py` only moves a camera as far as its mount allows. Until the
mounts are measured, run it with an assumed range and check only what it asks for:

```bash
python optimize_cameras.py --mounts configs/mounts_small.yaml   # +-15 deg, the default
python optimize_cameras.py --mounts configs/mounts_wide.yaml    # +-45 deg
```

`moves.md` then says, per camera, e.g. "turn 15 deg right, tilt 15 deg up":
try exactly that on the camera. If it can't, mark it (`fixed: true`, or a
smaller `pan` / `tilt_change`) and run again.

To measure a mount properly, per camera, into `configs/mounts.yaml`:

| what | key | how to measure |
|---|---|---|
| pan range | `pan: [lo, hi]` | turn the bracket left/right to its stops, degrees from where it points now |
| tilt range | `tilt: [lo, hi]` or `tilt_change: [lo, hi]` (from the current tilt) | lowest and highest tilt it holds (0 level, 90 straight down), including where the body or cable hits the wall/ceiling |
| position range | `along: [[x,y,z],[x,y,z]]` | if it can move (rail, clamp, new holes, cable reach to its host), the ends of that stretch in the room frame; leave out if it can only turn |
| don't touch | `fixed: true` | cameras someone depends on as they are |

Roll is kept at 0 (image upright). The lens is not a variable: each camera keeps
its calibrated intrinsics.

The search: each round, every camera is tried over its whole range (coarse
grid, then finer around the best) with the others held where they are, and the
one change that helps most is applied. It stops when no change adds
`--min-gain` of the volume (each move means re-calibrating that camera) or after
`--max-moved` cameras. Output: `moves.md` (per moved camera: heading, tilt,
position, and the floor/wall point to centre in the image when aiming it by
hand), `cameras.yaml` with the new poses, and before/after coverage reports.

## Measuring a mount's range from video

While a camera is turned by hand, its own stream shows how far it turned:
with the pivot close to the lens, two frames differ by a pure rotation, so
matched features give the angle without knowing the scene.

1. Lights on (the cameras do not brighten a dark room). Something with
   texture in view helps; bare wall alone does not.
2. Start recording the camera's own stream (camera hosts from the wiki):

   | cameras | stream |
   |---|---|
   | 3, 6, 8, 10, 12, 13, 15, 16 | `rtsp://192.168.1.13:8554/cam_N` |
   | 1, 2, 4, 5, 7, 9, 11, 14 | `rtsp://192.168.1.56:8554/cam_N` |

   ```bash
   ffmpeg -rtsp_transport tcp -i rtsp://192.168.1.56:8554/cam_2 -c copy cam_2_turn.mp4
   ```
3. Hold still 2 s. Loosen the bracket, turn **slowly** (about 5 s to each stop):
   full left, full right, back to the start; full down, full up, back to the
   start. Hold still 2 s, press `q`.
4. `python measure_rotation.py cam_2_turn.mp4 --camera cam_2 --cameras configs/cameras.yaml`

It prints the pan and tilt range (and writes `mount.yaml` to paste into
`configs/mounts.yaml`, 2 deg off each end), `rotation.png`, and a per-frame
`rotation.csv`.

- Angles are from the pose in `--cameras`, so the video must start with the
  camera where it was calibrated. With `cameras_nominal.yaml` the focal length
  is a guess and the angles scale with it.
- "end vs start" is the check: if you turned back to the start, it should be
  well under 1 deg. Larger means drift (too fast, blur, or a featureless view).
- Turning moves the camera: put it back to its marks, or re-calibrate it after.
- Tested on synthetic 1080p video with a known turn: range recovered within
  0.1 deg. A bracket whose pivot sits far from the lens adds parallax, small at
  room distances; not tested on the real cameras yet.

## What the nominal numbers mean

Nothing yet. The nominal layout guesses the room size (8 x 10 m), the spacing
along each wall, the lens (fx 1160 px = 4 mm on the IMX392's 3.45 um pixels,
about 79 x 50 deg), puts 13-16 on the bottom wall, and leaves out the offsets
of 4 and 6. It is there so the tool runs end to end, and it shows the kind of
gap to look for: with every camera aimed at the room centre, the corners next to
the cameras are seen by fewer than two of them at head height.

## Not modelled

People and robots blocking views (the reason to want k >= 2 and redundancy),
lens vignetting and blur, lighting. Obstacles are axis-aligned boxes.
