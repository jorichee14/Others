# ED305 camera calibration: the guide

Everything needed to calibrate the ED305 cameras (and any other RTSP camera) with the dashboard:
**intrinsics** (each camera's lens) and **extrinsics** (where each camera is, in the ZED's map).
All options: [SETTINGS.md](SETTINGS.md). Older material (the three-laptop way, floor-tag extrinsics):
[README.md](README.md).

```
1 · Intrinsics   board A (9 x 7)   every camera alone      -> results/intrinsics/cam_N.json
2 · Extrinsics   board E (4 x 4)   each camera + the ZED    -> results/extrinsics_ref/cam_N.json
```

Both happen in one web page (`server.py`) that several people open at the same time from a phone
or laptop. It runs either on a Windows PC on the ED305 network (intrinsics only) or on the robot's
Jetson (intrinsics and extrinsics; the ZED is read straight from ROS there).

---

## 1. Setup (once)

### Code

Repository `https://github.com/jorichee14/Others`, branch `digital_twin_v1`, folder
`ed305_cameras_calibration`. Update with `git pull` in the repository folder.

### On the Jetson (robot)

```bash
cd ~/workspaces
git clone -b digital_twin_v1 https://github.com/jorichee14/Others.git
sudo apt install -y python3-venv
python3 -m venv --system-site-packages ~/workspaces/calib-env
source ~/workspaces/calib-env/bin/activate
pip install --upgrade pip
pip install "opencv-contrib-python==4.10.0.84" "numpy<2" "scipy>=1.11,<1.14" flask pyyaml
```

`--system-site-packages` keeps ROS 2 (`rclpy`, `tf2_ros`) importable; the pinned versions keep numpy
below 2 so ROS keeps working. Check:

```bash
source /opt/ros/$ROS_DISTRO/setup.bash && source ~/workspaces/calib-env/bin/activate
python -c "import rclpy, tf2_ros, cv2, flask, scipy, yaml; print('ok', cv2.__version__, hasattr(cv2.aruco, 'CharucoDetector'))"
```
-> `ok 4.10.0 True`. Every new terminal: the two `source` lines first (or add them to `~/.bashrc`).

### On a Windows PC (optional, intrinsics only)

1. Install Git and Miniforge (PowerShell): `winget install -e --id Git.Git` and
   `winget install -e --id CondaForge.Miniforge3`.
2. In VS Code's terminal, once: `& "$env:USERPROFILE\miniforge3\Scripts\conda.exe" init powershell`
   and `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then reopen the terminal (it shows `(base)`).
3. ```
   git clone -b digital_twin_v1 https://github.com/jorichee14/Others.git
   cd Others\ed305_cameras_calibration
   conda create -n calib python=3.11 -y
   conda activate calib
   pip install -r requirements.txt
   ```
4. On Windows the command is **`python`**, not `python3` (that one is a different Python without the packages).
5. When Windows asks, allow Python on **private networks** so others can open the page.

### Boards (`config/boards.yaml`)

| Board | Used for | Layout |
|---|---|---|
| **A** | intrinsics | 9 x 7 squares, DICT_4X4_50, squares 24.5 mm, markers 18 mm (A4 print) |
| **E** | extrinsics | 4 x 4 squares, DICT_4X4_50, squares 12 cm, markers 9 cm |

- Mount each board **flat and rigid** (foam board, clipboard). A bent board gives wrong results.
- **Measure** a printed square and marker and write the real sizes (metres) into `boards.yaml`.
  They do not matter for intrinsics, they set the scale of every position in extrinsics.
- If the page finds **no corners** on a board, `squares_x` and `squares_y` are the wrong way round
  for that board: swap them.
- Several people may use identical copies of a board at the same time.

### Cameras (`config/cameras.yaml`)

The 16 ED305 cameras, `rtsp://192.168.1.13:8554/cam_N` (cams 3 6 8 10 12 13 15 16) and
`rtsp://192.168.1.56:8554/cam_N` (cams 1 2 4 5 7 9 11 14). The computer running the dashboard
must reach these addresses (be on the ED305 network). Other cameras: `--url N=rtsp://...`.

---

## 2. Start the dashboard

On the Jetson (both modes):
```bash
source /opt/ros/$ROS_DISTRO/setup.bash && source ~/workspaces/calib-env/bin/activate
cd ~/workspaces/Others/ed305_cameras_calibration
python3 server.py --ref 20 --ref-topic /zed/zed_node/left/image_rect_color --ref-board E
```
Only intrinsics (Jetson: `python3`, Windows: `python`):
```bash
python3 server.py                     # all 16 ED305 cameras
python3 server.py --cams 3 5          # only these
```

Open `http://<computer-ip>:8080` (find the IP with `hostname -I` on the Jetson, `ipconfig` on
Windows). Type your name, pick board **A**, **Join**. The buttons at the top switch between **1 · Intrinsics** and
**2 · Extrinsics**; the mode is the same for everyone.

Most used options (all in [SETTINGS.md](SETTINGS.md)): `--cams` (which cameras), `--url N=rtsp://...`
(another camera), `--hz` (detections per second, lower on a slow computer), `--port`,
`--access-log` (show web requests in the terminal), `--still-s` (extrinsics, laggy network).

---

## 3. Intrinsics (mode 1)

Before: lock focus, zoom, exposure and resolution of the camera; any change later means
recalibrating it. The ED305 cameras have manual focus rings (no autofocus).

1. On the overview, tap a camera. **Record**.
2. Hold board **A** in front of it and move it slowly: left/right/up/down (x, y), near and far (z),
   turned in its plane (roll), tilted forward/back (pitch) and sideways (yaw), into the corners of the
   image too. A frame saves by itself each time the board has moved.
3. When the six bars are green and 20+ frames are saved, **Calibrate** (lens model: the default
   "Standard, full (k1 k2 p1 p2 k3)").
4. Check: **test error under 1 px** (under 0.5 px very good). **Rectified** shows the undistorted
   picture (straight edges straight). In **Saved frames**, remove frames with a high error (X) and
   Calibrate again.
5. **Stop recording**, next camera. Calibrated cameras get a green outline on the overview.

- ED305 cameras: work about 1.5 to 3 m from the camera (their focus is set for the room). A bigger
  board (A3/A2: `python3 make_boards.py --page A3`) makes it faster.
- **Recalibrate (new round)** keeps the old result and frames and starts `cam_N_2`, `_3`, ...
- Lens models: `full` (default, any normal lens), `ed305` (k1 k2 only, centred principal point, like
  the March 2026 calibration), `fisheye` (over ~120 degrees).
- A camera calibrated elsewhere: `--intrinsics N=file` (its `cam_N.json` or a ROS camera_info YAML).

---

## 4. Extrinsics with the ZED (mode 2, on the Jetson)

Every camera is placed in the ZED's map (`map_zed`): the ZED and a camera both see board E at the
same moment, which gives that camera's pose relative to the ZED, and the ZED's own pose in its map
(cuVSLAM) places it in the map.

Before:
- The ZED wrapper runs with positional tracking (TF `map_zed -> ... -> zed_left_camera_optical_frame`;
  check with `ros2 run tf2_ros tf2_echo map_zed zed_left_camera_optical_frame`).
- Every camera to place has its intrinsics (mode 1); its card shows **✓ intrinsics**. The ZED's
  intrinsics come from its camera_info by themselves.
- Start the dashboard with the Jetson command above (`--ref 20 --ref-topic ... --ref-board E`).

Steps:
1. **2 · Extrinsics**. The ZED (camera 20) has a blue outline: the reference.
2. Drive the robot so the ZED and a camera both see the board area. **Stop the robot.**
3. Hold board E **still** (about 1 to 2 s) where both see it: green dots in both pictures.
4. Move the board to another spot or angle, hold still again. Repeat.
5. After **3+ placements** the camera is **placed** (green outline, **✓ placed**). A placement seen
   by several cameras counts for all of them.
6. Drive to the next cameras and repeat. The robot may move between placements, never while
   you hold the board.

Check on a camera's page: **spread under ~10 mm / 0.5 deg** (how much the samples disagree) and a
plausible position (the ED305 cameras hang 2.5 to 2.7 m high). Large spread: bent board, wrong
intrinsics, or the ZED's tracking drifting; **Collect again** (new round, the old file is kept).

No clock synchronisation is needed between the robot and the cameras: samples are only taken while
the board and the robot have been still for `--still-s` seconds (0.6) in every view. On a laggy
Wi-Fi, hold longer and use `--still-s 1.0`.

Other ways (same method): from a PC without ROS, with the ZED streamed over RTSP and its pose served
by `robot/zed_pose_server.py` (README, "Extrinsics with the ZED"); entirely as a ROS node on the robot
without a web page: `radar_camera_calibration/network_charuco.py`
(`radar_camera_calibration/NETWORK_CHARUCO.md`).

---

## 5. Output

| File | What |
|---|---|
| `results/intrinsics/cam_N.json` | camera N's lens: `K`, `dist`, `model`, errors, frames used; also the March 2026 keys (`camera_matrix`, `dist_coeffs`, `fx fy cx cy`, `img_w img_h`, `intrinsic`, `distortion`). `cam_N_2.json`... = later rounds |
| `results/intrinsics/summary.csv`, `views.csv` | one row per camera / per saved frame (overview tables for Excel; close them in Excel while calibrating) |
| `data/<person>/intr/cam_N/` | the saved frames (`.jpg` + `.json` corners); `deleted/` = removed ones |
| `results/extrinsics_ref/cam_N.json` | camera N in `map_zed`: `T` (4x4, map <- camera), `translation`, `quaternion_xyzw`, `rpy_deg`, samples, placements, spread |
| `results/extrinsics_ref/network.json` | every placed camera in one file |

The page shows them too: a camera page's **Calibration result** (Download JSON), and the tables
under **Results of all cameras** on the overview. Keep `data/` and `results/`.

---

## 6. Troubleshooting

| Problem | Fix |
|---|---|
| A camera card says **offline** / "No picture" | the computer cannot read its stream: is it on the ED305 network? `ping 192.168.1.13`, test the address in VLC |
| The ZED (camera 20) is offline | the dashboard does not see the ZED topic: `ros2 topic hz /zed/zed_node/left/image_rect_color` in the same terminal; same `ROS_DOMAIN_ID` as the ZED wrapper (or run the dashboard in the same container) |
| No green dots on the board | wrong board selected (intrinsics: A, extrinsics: `--ref-board E`), or `squares_x`/`squares_y` swapped in `boards.yaml` |
| `ModuleNotFoundError: flask` on Windows | use `python`, not `python3`, with `(calib)` active |
| `Permission denied ... views.csv` | the CSV is open in Excel: close it; the calibration itself is saved anyway |
| Test error high | rigid board, sharper frames (move slower), remove bad frames, more varied poses |
| Extrinsics spread large | rigid board, robot still while holding, `--still-s 1.0`, Collect again |
| Jetson slow | fewer cameras (`--cams`), `--hz 2` |
| Page looks empty / stuck | reload it; the terminal shows the dashboard's messages |

---

## 7. Checking the tools (no cameras needed)

```bash
python3 tests/test_models.py         # lens models recover a known camera
python3 tests/test_synthetic.py      # the older room solver (floor tags)
python3 tests/test_ref_extrinsics.py # extrinsics with the ZED (pose server), uneven stream delays
python3 tests/test_ros_ref.py        # extrinsics with the ZED read from ROS
python3 server.py --fake             # the page with simulated cameras
```
Each prints `PASS` (1 to 3 minutes).
