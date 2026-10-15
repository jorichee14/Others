# Settings

Everything that can be changed in the calibration dashboard, from the command line, in the code
and in the config files. Defaults are what you get with plain `python server.py`.

## 1. Command line: `python server.py ...`

| Option | Default | What it does |
|---|---|---|
| `--cams 1 2 17` | every camera in `config/cameras.yaml` | which cameras to open |
| `--url N=rtsp://...` | none | add camera N, or replace its address; repeat for several cameras |
| `--intrinsics N=FILE` | none | camera N is already calibrated: K and distortion from FILE (this folder's `cam_N.json`, the March 2026 `calibration/data/outputs/results/cam_N.json`, or a ROS `camera_info` YAML); rectified view right away |
| `--model full\|ed305\|fisheye` | `full` | lens model preselected on every camera page (see [Lens models](#lens-models)) |
| `--port 8080` | `8080` | port of the web page |
| `--host 0.0.0.0` | `0.0.0.0` | `0.0.0.0`: other PCs can open the page; `127.0.0.1`: only this PC |
| `--out data` | `data` | where saved frames go |
| `--results results` | `results` | where calibrations go |
| `--hz 4` | `4` | board detections per second per camera; higher feels faster, uses more CPU |
| `--workers N` | CPU cores - 1 | how many cameras are processed at the same time |
| `--fake` | off | simulated cameras, to try the page without any camera |
| `--ref N` | none | extrinsics mode places every camera relative to camera N, the reference (the ZED streamed with `ros_image_to_rtsp.py`); see README "Extrinsics with the ZED" |
| `--ref-pose URL` | none | with `--ref`: the ZED's pose from `robot/zed_pose_server.py` (`http://<robot>:8765/pose`); cameras are then placed in the ZED map and the robot may move between board placements |

Examples:

```
python server.py                                                       # the 16 ED305 cameras
python server.py --url 17=rtsp://192.168.51.94:8554/mobile_2/color     # ... plus the robot
python server.py --cams 17 --url 17=rtsp://192.168.51.94:8554/mobile_2/color --hz 6
python server.py --cams 17 --url 17=rtsp://... --intrinsics 17=camera_info.yaml
```

`calibrate_intrinsics.py` (the same fit without the dashboard) takes `--data`, `--results`,
`--cams`, `--model` and `--rational` (8 coefficients, with `--model full`).

## Lens models

Chosen on the camera page (dropdown above Record/Calibrate) for each Calibrate; `--model` only sets
what is preselected.

| Model | Fits | For |
|---|---|---|
| `full` (default) | k1 k2 p1 p2 k3, principal point estimated | normal lenses: the ED305 Baslers, the RealSense |
| `ed305` | k1 k2 only (p1 = p2 = k3 = 0), principal point fixed at the image centre, as the March 2026 calibration (`cal_params.py`) | comparing with the March numbers |
| `fisheye` | OpenCV fisheye (equidistant), k1 k2 k3 k4 | lenses wider than about 120 degrees |

## 2. In the code (edit, then restart the server)

### `server.py`, top of the file

| Name | Value | Meaning |
|---|---|---|
| `MIN_CORNERS` | `12` | fewest board corners for a frame to count as "board seen" |
| `ACTIVE_S` | `60` | seconds after which a person whose page is closed counts as gone |
| `STILL_PX`, `STILL_S` | `2.0` px, `0.8` s | extrinsics: the board is still when it moves less than 2 px for 0.8 s |
| `MOVE_PX` | `25` px | extrinsics: movement needed before the next event |
| `VIEW_S` | `4` | seconds a camera keeps its fast refresh after someone looked at it |

### `server.py`, function `process`: when a frame is saved

A frame is saved when the board has moved more than **6 %** of the image width (`0.06 * w`) or its
size changed by more than **12 %** (`0.12 * prev[1]`) since the last saved frame. Raise them to save
fewer, more different frames.

### `calib/intr.py`: the calibration

| Name | Value | Meaning |
|---|---|---|
| `DEFAULT_MODEL` | `"full"` | lens model used when none is given |
| `MIN_VIEWS` | `20` | calibrate frames needed before "Press Calibrate" |
| `TEST_EVERY` | `5` | every 5th saved frame is a test frame (never used in the fit) |
| `TARGET` | x `0.6`, y `0.6`, z `log(2.5)`, roll `60`, pitch `50`, yaw `50` | range each coverage bar needs to be full (x, y: fraction of the image; z: near/far ratio 2.5; angles in degrees) |
| outlier rule in `fit_camera` | `max(2 * med, 0.5)` | a calibrate frame with an error above twice the median (and above 0.5 px) is dropped from the fit |

Status (function `status`): `OK` needs test error under **1.0 px** (and not much above the
calibrate error), calibrate error under **0.6 px** and all six coverage bars full; otherwise `CHECK: ...`
says which one failed.

### `calib/streams.py`: reading the streams

| What | Value |
|---|---|
| check that the camera address answers, before opening | 2 s (`reachable(url, timeout=2.0)`) |
| open and read timeouts | 8 s (`CAP_PROP_OPEN_TIMEOUT_MSEC`, `CAP_PROP_READ_TIMEOUT_MSEC`) |
| RTSP transport | TCP (`OPENCV_FFMPEG_CAPTURE_OPTIONS`) |

### `web/dashboard.html`: the page

| What | Value |
|---|---|
| status refresh | every 0.7 s (`setTimeout(poll, 700)`) |
| live picture on a camera page | every 0.3 s |
| thumbnails on the overview | every 1.5 s |

## 3. Config files (`config/`)

| File | Contents |
|---|---|
| `cameras.yaml` | camera number -> stream address (the 16 ED305 cameras). Add the robot here (`17: rtsp://...`) instead of passing `--url` every time |
| `boards.yaml` | the board: squares across x down (9 x 7), dictionary (`DICT_4X4_50`), black square and marker size in metres (24.5 mm, 18 mm), and the floor tag type and size |
| `floor_markers.yaml` | floor tag positions in the room, for extrinsics |
| `assignments.yaml` | which person does which cameras (three-laptop way only) |

## 4. Where the output goes

| Path | What |
|---|---|
| `results/intrinsics/cam_N.json` | camera N's calibration: K, distortion, model, errors, frames used; also the March 2026 keys (`camera_matrix`, `dist_coeffs`, `fx` `fy` `cx` `cy`, `img_w` `img_h`, `intrinsic`, `distortion`) |
| `results/intrinsics/cam_N_2.json`, `_3` ... | later rounds (Recalibrate); earlier rounds are kept |
| `results/intrinsics/summary.csv` | one row per camera, all cameras |
| `results/intrinsics/views.csv` | one row per saved frame, all cameras: split, used, error, board pose |
| `data/<person>/intr/cam_N/` | saved frames of camera N: `.jpg` picture + `.json` detected corners |
| `data/<person>/intr/cam_N/deleted/` | frames removed with the X button (kept, not used) |
| `results/extrinsics/` | room frame of all cameras (extrinsics round) |
| `results/extrinsics_ref/cam_N.json` | with `--ref`: camera N's pose relative to the reference camera, or in the ZED map with `--ref-pose` (`T`, translation, quaternion, roll-pitch-yaw, samples, placements, spread); `cam_N_2.json` ... after "Collect again" |
| `results/extrinsics_ref/network.json` | with `--ref`: every placed camera in one file |
