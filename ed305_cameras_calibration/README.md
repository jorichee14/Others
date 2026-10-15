# ED305 camera calibration (16 cameras, 3 people)

Intrinsics and room-frame extrinsics for the 16 Basler cameras in ED305, captured from the
RTSP streams so nothing stops the live camera wall and three people can work at the same time.

Result: `results/intrinsics/cam_N.json` (K, distortion), `results/extrinsics/cameras.json`
(R, t of every camera in the room frame, camera positions) and `results/extrinsics/report.md`,
plus CSV files with every view, event and camera (`x y z roll pitch yaw`, errors, calibrate/test
split) so the calibration can be re-checked at any time with `evaluate.py`.

**Calibrate and test sets.** Every 5th saved view (intrinsics) and every 5th still-board event
(extrinsics) is a TEST sample. Test samples are never used in the fit; the error on them
(`rms_test_px`, `floor_err_test_cm`) is the honest quality number. The split is recorded in
`results/intrinsics/views.csv` and `results/extrinsics/events.csv`.

All options and settings: [SETTINGS.md](SETTINGS.md).

## Try it now with one RTSP camera (any stream, e.g. the robot)

On the PC that will run the dashboard (it must reach the stream and the other people must reach it):

```
cd ed305_cameras_calibration
pip install -r requirements.txt
python server.py --cams 17 --url 17=rtsp://<robot-ip>:8554/mobile_2/color
```

1. Open `http://<this pc's ip>:8080` on your laptop; the other two open the same address on
   theirs (same network). Anyone can tap camera 17 and watch it live without joining.
2. Board: any ChArUco board whose layout is in `config/boards.yaml` (the A1 boards A, B, C, or
   print one with `python make_boards.py --page A3` and write the printed square size in).
3. Type your name, pick your board, **Join**, then **Record** on camera 17. Move the board
   through the image (corners and edges too, near and far, tilted). Frames save by themselves
   when the board has moved; the six bars, the frame count and the coverage map fill up.
4. **Calibrate** (10+ frames, better 20+ with all bars green). K, distortion and the
   calibrate/test error appear; `results/intrinsics/cam_17.json` is written.
5. **Rectified** or **Both** shows the undistorted live view right away.
6. **Frames** panel: every saved frame with its detected corners (green) and where the
   calibration puts them (red) with the error. Delete bad frames (high error, blurred) and
   press Calibrate again.

Already calibrated? The rectified view is there from the start:
- a camera calibrated here before: `results/intrinsics/cam_N.json` is loaded at startup;
- calibrated elsewhere: `--intrinsics 17=camera_info.yaml` takes a ROS camera_info
  (`ros2 topic echo --once /<camera>/camera_info > camera_info.yaml`, or a `camera_calibration`
  YAML), this folder's `cam_N.json`, or the wiki's `calibration/data/outputs/results/cam_N.json`.
  If the file is for another resolution than the stream, K is scaled to the stream.

Several streams at once: repeat `--url` (`--cams 17 18 --url 17=... --url 18=...`), or add them
to `config/cameras.yaml`. A test run over RTSP of a simulated 1280x720 camera (fx 900,
k1 -0.25, k2 0.08) gave fx 900.2, fy 900.0, k1 -0.250, k2 0.084 and 0.26 px test error.

## All 16 ED305 cameras: one command

On a PC on the ED305 network (lab Wi-Fi or VPN):
```
python server.py
```
Without `--cams` it opens every camera in `config/cameras.yaml` (cam_1 ... cam_16); add the robot
with `--url 17=rtsp://<robot-ip>:8554/mobile_2/color`. Reading 16 streams of 1920x1080 wants a
PC with 8+ cores. The overview shows every camera; calibrated ones have a green outline.

**Recalibrate** (camera page) starts a new round without deleting anything: new frames go to
`data/<person>/intr/cam_N_2/` and Calibrate writes `results/intrinsics/cam_N_2.json` (then
`_3`, ...). Round 1 (`cam_N`, `cam_N.json`) stays as it was. Everything uses the newest round
that has a result; until the new round is calibrated the previous calibration stays in use.

## Extrinsics with the ZED (reference camera)

Every camera placed relative to the robot's ZED, in the ZED's map, from the dashboard's **2 · Extrinsics**
mode. The same method as `radar_camera_calibration/network_charuco.py` (which runs on the robot under
ROS instead); see `radar_camera_calibration/NETWORK_CHARUCO.md` for how it works.

On the robot (ROS 2 sourced, ZED wrapper running with positional tracking):
```
python3 ros_image_to_rtsp/ros_image_to_rtsp.py --topic /zed/zed_node/left/image_rect_color --url rtsp://localhost:8554/zed/left
python3 ed305_cameras_calibration/robot/zed_pose_server.py        # http://<robot>:8765/pose
ros2 topic echo --once /zed/zed_node/left/camera_info > zed_left_info.yaml
```
(MediaMTX running on the robot for the RTSP stream, as for `mobile_2/color`; copy `zed_left_info.yaml`
to the dashboard PC.) On the dashboard PC, with every other camera's intrinsics already done:
```
python server.py --url 20=rtsp://<robot-ip>:8554/zed/left --intrinsics 20=zed_left_info.yaml \
                 --ref 20 --ref-pose http://<robot-ip>:8765/pose
```
Press **2 · Extrinsics**. Hold the board **still** where the ZED and a camera both see it (green dots
in both), move it, hold again; the robot may drive between placements but must stand still while you
hold. A camera is **placed** (green outline) after 3+ placements; its pose is in
`results/extrinsics_ref/cam_N.json` and all of them in `network.json`. "Collect again" on a camera
page starts a new round (`cam_N_2.json`), keeping the old file. Without `--ref-pose` the poses are
relative to the ZED itself and the robot must not move at all.

## Lens models

Picked on the camera page (or `calibrate_intrinsics.py --model ...`) before Calibrate:

| model | what | for |
|---|---|---|
| `full` (default) | OpenCV pinhole, k1 k2 p1 p2 k3, principal point estimated | any normal lens: the ED305 Baslers, RealSense |
| `ed305` | k1 k2 only, principal point at the image centre: how the March 2026 ED305 calibration (`cal_params.py`) did it | comparing with that calibration |
| `fisheye` | OpenCV fisheye (equidistant), k1 k2 k3 k4 | lenses over about 120 degrees |

Every result `cam_N.json` also has the keys of the March 2026 files (`camera_matrix`, `dist_coeffs`,
`fx fy cx cy`, `img_w img_h`, `intrinsic`, `distortion`), so tools that read those read these.
The room extrinsics use the standard models only (a fisheye camera is left out there).

## The whole process in one list

1. Print boards A, B, C (A1) and 12 floor tags (A4): `python make_boards.py`. Mount the boards flat.
2. Measure a printed square and a tag edge; write them into `config/boards.yaml`.
3. Tape the tags to the floor, measure each tag centre from the two walls, write them into
   `config/floor_markers.yaml`. Lock exposure, gain and focus on every camera.
4. On the lab PC (scoop env, `pip install flask` once): `python server.py`. Everyone opens the
   address it prints on a phone or laptop and joins with their name and board.
5. Intrinsics: tap a camera, press Record, move the board until the six bars are green, press Calibrate,
   next camera. All 16 cameras: about 2 hours for three people.
6. Extrinsics: press "2 Extrinsics", hold the boards still where cameras overlap until 150+
   events, press "Solve extrinsics". About 1 to 2 hours.
7. Read the table: z between 2.5 and 2.7 m, test rms under 1 px, floor error under 5 cm.
   Optional: `python verify.py --tag 11 --x 2.00 --y 4.00` at a few measured spots.
8. Keep `data/`, `results/` and the two config files. Later: `python evaluate.py` after a
   short new recording tells whether a camera moved.

Two ways to run it:

- **Dashboard (recommended): one PC, everyone on a browser.** `python server.py` on a PC in the
  lab reads all 16 streams; each person opens `http://<that pc>:8080` on a phone or laptop,
  picks a name, board and camera, and sees the live view, the frame count and the six coverage
  bars. All frames and results land on that PC. Try it without cameras: `python server.py --fake`.
- **Three laptops with `capture.py`** (the same logic in an OpenCV window, one person per laptop,
  folders merged afterwards). Described further down.

## Setup (once)

```
pip install -r requirements.txt
```
Be on the ED305 network (lab Wi-Fi or VPN); `http://192.168.1.188:3000/cameras` must open.
Stream addresses are in `config/cameras.yaml` (from the wiki). The dashboard PC wants 8+ cores
(each camera runs one board detection per board in the room, 4 times a second).

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

## Calibration day with the dashboard

On the lab PC, in this folder: `python server.py`. It prints the address; everyone opens it.

1. **Join.** Type your name, pick your board (A, B or C: one per person) and press Join.
2. **Intrinsics.** Tap your camera in the grid and press **Record** (a camera can be recorded by one person at a time;
   camera groups per person are in `config/assignments.yaml`). Walk the board in front of it.
   Frames save by themselves when the board has moved; the page shows the count and the six
   bars x y z roll pitch yaw (what each one wants is written under the live view). When all
   bars are green and 20+ frames are saved the page says READY: press **Calibrate**. It shows
   `rms calib / test`; want calib under 0.6 px and test under 1 px, else keep moving the board
   and press Calibrate again. Check the Frames panel (delete frames with a high error) and
   the Rectified view. Then press Stop recording and tap your next camera. The grid dots show every camera's
   state (grey none, yellow some frames, green ready, blue calibrated).
3. **Extrinsics.** When all 16 are calibrated, press **2 Extrinsics** (one mode for everyone).
   Every camera now watches every joined board. Hold your board still for a second where two
   or more cameras see it: the event counter goes up. Move, hold again, cover every overlap.
   Aim for 150+ events in total. Then press **Solve extrinsics**: the room-frame table
   (x y z roll pitch yaw, calibrate/test error, floor error) appears at the bottom.
4. Everything is in `data/` and `results/` on that PC, the same files as below.

## Calibration day with three laptops (no dashboard)

Everyone works on their own laptop in their own output folder (`data/<person>/`).
Camera groups per person are in `config/assignments.yaml`.

**Round 1, intrinsics (about 2 hours, all three at once).**
```
python capture.py --person Jorich  --board A --mode intr --cams 1 2 3 4 13 14
python capture.py --person Ralph   --board B --mode intr --cams 5 6 7 8 15 16
python capture.py --person Britney --board C --mode intr --cams 9 10 11 12
```
Walk your board in front of your cameras. Frames save by themselves when the board is well
seen and has moved. Each camera tile shows six bars, like the X Y Size Skew bars of the ROS
`cameracalibrator`, for the range of board poses saved so far:

| bar | what to do to fill it |
|---|---|
| x, y | move the board to the left, right, top and bottom of the image |
| z | hold it close (fills the image) and far (small) |
| roll | turn the board in its own plane, about 60 degrees between extremes |
| pitch | tilt it forward and back (top edge towards / away from the camera) |
| yaw | tilt it left and right (one side towards the camera) |

When all six bars are green and 20+ frames are saved the tile says **READY: press c**. `c`
calibrates every camera that has frames (the CALIBRATE button): it fits on the calibrate views,
measures on the held-out test views and shows `rms calib / test` on the tile. Want rms under
0.6 px and test under 1.0 px. If not, keep moving the board (more tilted and corner views) and
press `c` again. The fit is written to `results/intrinsics/cam_N.json` straight away.

The same solve from saved frames, for all cameras at once, or to redo it later:
```
python calibrate_intrinsics.py --data data --cams 1 2 3 4 13 14
```
It writes `results/intrinsics/summary.csv` (per camera: errors, K, distortion, coverage of
each bar, status) and `views.csv` (per view: split, used or dropped, error, board x y z roll
pitch yaw).

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
reprojection error per camera on the calibrate events and on the held-out test events, plus
the floor-point error of the floor tags seen in test events (want under 5 cm), and writes
`results/extrinsics/report.md`, `cameras.csv` (x y z roll pitch yaw per camera), `events.csv`
(board pose and error per event) and `tags.csv`.

**Verify (about 1 hour).** Put a floor tag at a measured spot and run
```
python verify.py --tag 11 --x 2.00 --y 4.00
```
at 10 spots across the room. Target: under 5 cm on the floor plane, which is what the BEV
person detection needs. Record the numbers in the report.

## Re-evaluation later (any time, 15 minutes)

To check whether a camera has been bumped or drifted, record a short new session into a new
folder (any person name, intr and/or extr mode, no need for full coverage) and evaluate it
against the stored calibration without refitting:
```
python capture.py --person check --board A --mode extr --out data_check
python evaluate.py --data data_check --all
```
It prints, per camera, the new reprojection error next to the stored test error and the floor
error, and flags `CHECK` when a camera no longer fits. Results go to
`results/evaluation/<date>/cameras.csv` and `details.csv`. `python evaluate.py --data data`
(without `--all`) re-scores the original test sets only.

## What to keep

`data/` (raw frames and detections, so anything can be re-run), `results/` (including the CSVs
with the calibrate/test split), the filled-in `config/floor_markers.yaml` and
`config/boards.yaml`. Re-run round 2 after any camera is bumped.

Conventions in the CSV and JSON files: room frame x, y along the walls, z up, metres. A
camera's `roll pitch yaw` are of its pose in the room, `R_wc = Rz(yaw) Ry(pitch) Rx(roll)`,
degrees. A view's `x y` are the board centre in the image (0 to 1), `z` its distance along the
optical axis, `roll` its in-plane rotation and `pitch`, `yaw` its tilt about the image axes.

## Files

| File | What |
|---|---|
| `SETTINGS.md` | every option, setting, config file and output path in one place |
| `server.py`, `web/dashboard.html` | one-PC dashboard: everyone captures and calibrates from a browser |
| `capture.py` | capture frames from RTSP in an OpenCV window, intr or extr mode |
| `calibrate_intrinsics.py` | per-camera K and distortion |
| `calibrate_extrinsics.py` | all cameras in the room frame (bundle adjustment with floor tags), test on held-out events |
| `evaluate.py` | re-score a stored calibration on test sets or on a new recording, no refit |
| `verify.py` | triangulate a tag live and compare with a measured spot |
| `make_boards.py` | printable boards and floor tags |
| `calib/intr.py`, `calib/extr.py` | the fitting, split and evaluation code shared by the scripts |
| `config/` | camera URLs, boards, floor tag survey, who does which cameras |
| `tests/test_synthetic.py` | end-to-end check on a simulated room, no cameras needed |

Time sync is not part of this: the cameras are software-synced on each host (wiki, "同步錄影").
For calibration the board is held still, so sync does not matter.
