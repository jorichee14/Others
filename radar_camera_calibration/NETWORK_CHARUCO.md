# ZED + camera network extrinsics (`network_charuco.py`)

The same method is built into the calibration dashboard (`ed305_cameras_calibration`, "Extrinsics with
the ZED" in its README): there the ZED is streamed over RTSP and its pose served by
`robot/zed_pose_server.py`, and everything is done from a browser. This ROS node is the alternative
that runs entirely on the robot.

`general_charuco.py` gives the extrinsic of ONE camera pair (the ZED and one other ROS camera).
`network_charuco.py` does the same for the ZED and MANY cameras at once, including cameras that are
not on ROS (the 16 ED305 cameras, streamed over RTSP), and places every camera in the ZED map.

```
camera A = the ZED        ROS 2: rectified left image + camera_info, its pose in map_zed (TF, cuVSLAM)
cameras B = RTSP cameras  ED305 cam_1 ... cam_16, the robot's RealSense via ros_image_to_rtsp.py, ...
                          intrinsics: ed305_cameras_calibration/results/intrinsics/cam_N.json (the dashboard)
board                     ed305_cameras_calibration/config/boards.yaml (board A), the same for all cameras
```

Each time the board is held **still** where the ZED and a camera both see it, one sample is taken:

```
T_map_B = T_map_ZED · T_ZED_board · inv(T_B_board)
```

The ZED's own pose at that moment (`T_map_ZED`, from TF) is part of every sample, so the robot may
drive around between board placements. Samples are pooled and averaged per camera; a camera
**locks** when the average is stable (as in `general_charuco.py`) and has samples from at least 3
board placements. On lock its pose is written to `network_extrinsics/cam_N.json` and published as a
static TF `map_zed -> ed305_cam_N`.

RTSP frames have no capture timestamp, so A and B are paired by **stillness**, not by time: only
while the board is still in both images. Hold it still for about a second each time.

## Before

1. **Intrinsics of every camera B**: calibrate them in the dashboard (`ed305_cameras_calibration`,
   standard 5-coefficient model). Copy its `results/` folder to the robot, or pass its path with
   `intrinsics_dir`. Fisheye calibrations are not supported here.
2. **ZED**: the ZED wrapper running with positional tracking (cuVSLAM), so TF has
   `map_zed -> zed_left_camera_optical_frame` (change `world_frame` / `cam_a_map_frame` if your tree
   names them differently). The ZED's rectified image needs no extra intrinsics: they come from
   `camera_info`.
3. **Network**: the computer running the node (the robot) must open the RTSP addresses
   (e.g. `rtsp://192.168.1.13:8554/cam_3`). Check one in VLC or `ffprobe` from the robot.
4. **Board**: large enough for both. The ED305 cameras see it from 3 to 6 m and the ZED (720p) from
   1.5 to 3 m; squares should be at least ~15 px in both images. The A4 board (24.5 mm squares) is
   too small for that; print an A2 or A1 board with `python make_boards.py --page A2` in
   `ed305_cameras_calibration`, mount it flat and rigid, measure it, and write it into `boards.yaml`.
5. **Python on the robot**: `pip install "opencv-contrib-python>=4.8" scipy pyyaml` (the ChArUco
   detector of OpenCV 4.7+), plus ROS 2 with `cv_bridge` and `tf2_ros`.

## Step 1: ZED + one camera (try it)

One camera first, e.g. ED305 cam 3, or the robot's RealSense streamed as RTSP:

```
python3 network_charuco.py --ros-args -p cameras:="3"
python3 network_charuco.py --ros-args -p urls:="17=rtsp://192.168.51.94:8554/mobile_2/color"
```

For two cameras **on the same robot** (ZED + RealSense) add `-p use_map:=false`: the samples are then
`T_ZED_B` directly (no map needed; the robot may move, the two cameras must not move relative to
each other).

Hold the board still where both see it, move it to another spot and angle, hold again. Every 2 s
the node prints per camera: samples, board placements, stability (t-std, rot-std) and what it sees
right now. After a few placements it prints `=== cam 3 LOCKED ===` with the pose.

## Step 2: the camera network

```
python3 network_charuco.py --ros-args -p cameras:="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16"
```

Drive the robot to a camera, hold the board where the ZED and that camera see it (3 or more
placements), move on. A placement seen by several ED305 cameras at once counts for all of them.
Cameras already locked are skipped. Stop with Ctrl-C at any time; what is there is saved.

## Output (`output_dir`, default `./network_extrinsics`)

| File | Contents |
|---|---|
| `cam_N.json`, `cam_N.yaml` | camera N in `map_zed`: `translation`, `quaternion_xyzw`, `rpy_deg`, `T` (4x4, map <- camera), `stage` (`locked` / `live`), `samples`, `board_placements`, `spread_mm` / `spread_deg` (how much the samples disagree), `reproj_a_px` / `reproj_b_px`. Same keys as `general_charuco.py`'s file, plus these |
| `network.json` | every camera in one file (rewritten every 2 s) |
| TF | static `map_zed -> ed305_cam_N` for every locked camera (`publish_tf`, `child_prefix`) |

A good camera: spread under ~10 mm / 0.5 deg and 3+ placements. A large spread means a bent board,
wrong intrinsics, or the ZED's map pose drifting; recalibrate that camera.

## Parameters (`--ros-args -p name:=value`)

| Parameter | Default | |
|---|---|---|
| `image_topic`, `info_topic` | `/zed/zed_node/left/image_rect_color`, `.../left/camera_info` | the ZED |
| `world_frame`, `cam_a_map_frame` | `map_zed`, `zed_left_camera_optical_frame` | map frame and the ZED's frame in TF |
| `use_map` | `true` | `false`: pool T_ZED_B (cameras on the same rig) |
| `cameras` | | ED305 camera numbers from `config/cameras.yaml`, e.g. `"1,2,3"` |
| `urls` | | other streams, `"17=rtsp://...,18=rtsp://..."` |
| `intrinsics_dir` | `../ed305_cameras_calibration/results` | where `intrinsics/cam_N.json` are |
| `board` | `A` | board name in `boards.yaml` |
| `rate` | `5` | detections per second |
| `min_corners`, `max_reproj_px` | `8`, `1.5` | a board pose is used only with enough corners and a low error |
| `still_s`, `still_mm`, `still_deg` | `0.6`, `3`, `0.3` | still = moved less than 3 mm / 0.3 deg over 0.6 s |
| `conv_window`, `conv_hold`, `conv_t_std_mm`, `conv_rot_std`, `conv_pool` | `15`, `10`, `3`, `0.2`, `30` | lock when stable, as `general_charuco.py` |
| `min_holds` | `3` | board placements needed before a lock |
| `output_dir` | `network_extrinsics` | |
| `publish_tf`, `child_prefix` | `true`, `ed305_cam_` | static TF on lock |

Test without ROS or cameras: `python3 tests/test_network_charuco.py` (simulated room, the robot
moving between placements; checks both cameras lock within 10 mm / 0.3 deg of the truth).
