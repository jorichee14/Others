# ED305 camera calibration (16 cameras, 3 people)

Intrinsics and room-frame extrinsics for the 16 Basler cameras in ED305, captured from the
RTSP streams so nothing stops the live camera wall and three people can work at the same time.

Result: `results/intrinsics/cam_N.json` (K, distortion), `results/extrinsics/cameras.json`
(R, t of every camera in the room frame, camera positions) and `results/extrinsics/report.md`.

## Setup (once, each laptop)

```
pip install -r requirements.txt
```
Be on the ED305 network (lab Wi-Fi or VPN); `http://192.168.1.188:3000/cameras` must open.
Stream addresses are in `config/cameras.yaml` (from the wiki).

## Before calibration day (half a day, one person)

1. Lock the camera settings and never touch them again (exposure, gain, focus). Any change
   after calibration invalidates it. Check sharpness with the wiki's `focus_score.py`.
2. `python make_boards.py --out print` makes three ChArUco boards (A, B, C: one per person,
   each with its own marker type so they can be in the room together) and 12 floor tags.
   Print the boards on A1 at 100% on rigid foam board; tags on A4. Measure a printed square
   and a tag edge and put the real sizes in `config/boards.yaml`.
3. Define the room frame: origin at a marked floor corner, X and Y along the walls, Z up.
   Tape the floor tags square to the walls, spread over the floor with more near the centre
   (all cameras aim at the point under cameras 10 and 11). Measure each tag centre with a laser
   distance meter from the two walls, or read it off the LiDAR room map, and write it into
   `config/floor_markers.yaml`. These tags are what ties every camera to the same room frame
   as the robots.
4. Each person checks their laptop can show their cameras: `python capture.py --person Me
   --board A --mode intr --cams 1 2` and press q.

## Calibration day

Everyone works on their own laptop in their own output folder (`data/<person>/`).
Camera groups per person are in `config/assignments.yaml`.

**Round 1, intrinsics (about 2 hours, all three at once).**
```
python capture.py --person Jorich  --board A --mode intr --cams 1 2 3 4 13 14
python capture.py --person Ralph   --board B --mode intr --cams 5 6 7 8 15 16
python capture.py --person Britney --board C --mode intr --cams 9 10 11 12
```
Walk your board in front of your cameras: near and far, tilted, in every part of the image.
The window counts saved frames per camera (target 40) and shows a 3x3 map of which image
regions still lack views (`.` empty, `#` covered). Frames save by themselves when the board
is well seen and has moved. Then, each person on their own cameras:
```
python calibrate_intrinsics.py --data data --cams 1 2 3 4 13 14
```
Want rms under 0.6 px per camera. If a camera is worse, capture more tilted and corner views.

**Round 2, extrinsics (about 2 hours, all three at once).**
Every laptop watches all 16 cameras; every person walks their own board:
```
python capture.py --person Jorich --board A --mode extr
```
Hold the board still for a second where two or more cameras see it; the tool saves the set
of frames and says "event N saved". Move, hold again. Cover every overlap: wall to wall across
the centre, wall to ceiling, along both ceiling rows. Aim for 150 or more events in total,
every camera in 20 or more. Keep the floor tags visible as much as possible (they are
recorded in every frame that sees them).

**Merge and solve (one person).** Copy everyone's `data/<person>` folders into one `data/`
(or point `--data` at the shared folder) together with `results/intrinsics`, then:
```
python calibrate_extrinsics.py --data data --results results
```
It prints the camera positions in the room frame (z should be 2.5 to 2.7 m) and the
reprojection error per camera, and writes `results/extrinsics/report.md`.

**Verify (about 1 hour).** Put a floor tag at a measured spot and run
```
python verify.py --tag 11 --x 2.00 --y 4.00
```
at 10 spots across the room. Target: under 5 cm on the floor plane, which is what the BEV
person detection needs. Record the numbers in the report.

## What to keep

`data/` (raw frames and detections, so anything can be re-run), `results/`, the filled-in
`config/floor_markers.yaml` and `config/boards.yaml`. Re-run round 2 after any camera is bumped.

## Files

| File | What |
|---|---|
| `capture.py` | capture frames from RTSP, intr or extr mode |
| `calibrate_intrinsics.py` | per-camera K and distortion |
| `calibrate_extrinsics.py` | all cameras in the room frame (bundle adjustment with floor tags) |
| `verify.py` | triangulate a tag live and compare with a measured spot |
| `make_boards.py` | printable boards and floor tags |
| `config/` | camera URLs, boards, floor tag survey, who does which cameras |
| `tests/test_synthetic.py` | end-to-end check on a simulated room, no cameras needed |

Time sync is not part of this: the cameras are software-synced on each host (wiki, "同步錄影").
For calibration the board is held still, so sync does not matter.
