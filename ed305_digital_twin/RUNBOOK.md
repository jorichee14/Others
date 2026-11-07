# ED305 runbook: from recordings to trust values and figures

Every command, in order. Run from `ed305_digital_twin/` with the conda env active
(`conda activate yolo`). The method is in [TRACKING.md](TRACKING.md), the detector in
[DETECTOR.md](DETECTOR.md).

```
once            once per calibration        per recording session                      for the paper
setup ─► detector ─► room frame ─────────► sync ─► track + trust + video ─► figures ─► schematics
                     make_room_cameras      motion_sync  room_track              make_figures  make_schematics
                                            └────────────── run_pipeline.sh ──────────────┘
```

## 0. Setup (once per machine)

```bash
conda create -n yolo python=3.11 -y && conda activate yolo
pip install torch torchvision          # Linux; on Windows add --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt -r requirements-yolo.txt
pip install cairosvg                   # optional: PDF/PNG schematics
python -m pytest -q tests              # all tests pass
```

## 1. Detector (once; done: `runs/detect/ed305_v1/weights/best.pt`)

```bash
python extract_frames.py ~/Downloads/yolo_20261105_10/cam_*.mkv --out data/overhead
python prelabel.py data/overhead --classes person
python show_labels.py data/overhead                                   # look before going on
python make_dataset.py data/overhead --val-cameras cam_3,cam_12 --val-frames 300
#   correct the 300 val frames (CVAT, or python fix_labels.py data/overhead)
python train_yolo.py data/overhead/data.yaml --name ed305_v1
yolo val model=runs/detect/ed305_v1/weights/best.pt data=data/overhead/data.yaml imgsz=1280
```

## 2. Cameras in the room frame (once per calibration)

Inputs: camera poses in the LiDAR map, the lens calibrations (`configs/intrinsics/`, in the
repository), the LiDAR map of the room.

```bash
python make_room_cameras.py \
  --extrinsics ~/workspaces/isaac_ros-dev/data/processed/20261103/mapping/cameras_ed305/cameras.yaml \
  --intrinsics configs/intrinsics \
  --floor-map ~/workspaces/isaac_ros-dev/data/processed/20261103/mapping/mapping/ed305_room_ceil.ply
```

Writes `configs/cameras_room.yaml`. Check the printout: floor tilt about 1 deg, camera heights
2.6-3.0 m, each camera looking at the part of the room it shows.

## 3. One recording session, end to end

One video per camera in the folder (move `.mp4` copies aside when `.mkv` are there):

```bash
REC=~/Downloads/yolo_20261105_10 START=60 END=180 OUT=out/run1 ./run_pipeline.sh
```

It runs, skipping what is already done:

| step | command it runs | output |
|---|---|---|
| sync | `python motion_sync.py $REC` | `cam_N_times.csv` next to each video (pair residuals in ms) |
| track + trust + video | `python room_track.py --rec $REC --calib configs/cameras_room.yaml --model … --reid yolo26s-reid.onnx --device 0 --video all --out $OUT` | `tracks.csv`, `persons.csv`, `persons_state.csv`, `boxes.csv`, `all_ids.mp4`, `summary.md` |
| figures | `python make_figures.py $OUT --calib configs/cameras_room.yaml` | `$OUT/figures/` |

Settings are environment variables (top of `run_pipeline.sh`): `MODEL`, `REID` (empty = no
appearance), `DEVICE`, `RATE`, `START`/`END`, `SENSORS` (sensor models and the door zone),
`EXTRA` (other sensors' measurement CSVs), `RESYNC=1`.

Single steps by hand:

```bash
python motion_sync.py ~/Downloads/yolo_20261105_10 --check-drift
python room_track.py --rec ~/Downloads/yolo_20261105_10 --calib configs/cameras_room.yaml \
  --model runs/detect/ed305_v1/weights/best.pt --reid yolo26s-reid.onnx --device 0 \
  --start 60 --end 180 --video all --out out/run1
python make_figures.py out/run1 --calib configs/cameras_room.yaml --persons 4 --frame-at 95
python mosaic.py ~/Downloads/yolo_20261105_10 --model runs/detect/ed305_v1/weights/best.pt --track --out out/all.mp4
```

## 4. Other sensors (radar, robots)

Each sensor writes measurements as CSV (`t,sensor,x,y,sxx,sxy,syy[,ident][,f0,...]`, room
frame) and gets an entry in a sensors YAML (detection probability, clutter, coverage, identity):

```bash
cp configs/sensors_example.yaml configs/sensors.yaml      # edit: radar field of view, door position
SENSORS=configs/sensors.yaml EXTRA="radar.csv robot_1.csv" REC=… ./run_pipeline.sh
```

## 5. Simulated room (no data needed)

```bash
python room_track.py --sim dense  --out out/sim_dense     # 16 cameras
python room_track.py --sim fused  --out out/sim_fused     # + radar + a robot
python room_track.py --sim sparse --out out/sim_sparse    # cameras with a blind band
python make_figures.py out/sim_sparse
```

Each prints IDF1, switches and tracks against the known identities.

## 6. Outputs

| file | one row per | columns |
|---|---|---|
| `persons_state.csv` | person, time step | `x y vx vy` (m), `sxx sxy syy` covariance, `sigma_m`, `seen`, `age_s`, `views`, `id_conf`, `link_conf` |
| `boxes.csv` | camera detection | `t camera x0 y0 x1 y1 person label_conf det_score` |
| `tracks.csv` | track update | `track t x y vx vy` |
| `persons.csv` | track | which person it belongs to (duplicates merged) |
| `measurements.csv` | measurement | position, covariance, feature, the track it went to (empty = clutter) |
| `all_ids.mp4` | time step | every camera with person ids + the floor plan |

## 7. Figures and schematics

| what | made by | for a paper | for slides, docs |
|---|---|---|---|
| floor paths (small multiples) | `make_figures.py` | `figures/floor_paths.pdf` | `.png` |
| trust over time (small multiples) | `make_figures.py` | `figures/trust_timeline.pdf` | `.png` |
| camera mosaic frame | `make_figures.py --frame-at <s>` | `figures/mosaic_frame.png` | same |
| framework schematic | `make_schematics.py` | `docs/figures/framework.pdf` | `.png`, `.svg` |
| roadmap schematic | `make_schematics.py` | `docs/figures/roadmap.pdf` | `.png`, `.svg` |

LaTeX:

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/framework.pdf}
  \caption{Any sensor in, one calibrated belief out, any task on top.}
  \label{fig:framework}
\end{figure}
```

PDF figures are vector (sharp at any size, fonts embedded); use PNG only for the camera
frame and for slides. To change a schematic, edit its labels in `make_schematics.py` and run
it again, or open the `.svg` in Inkscape, Illustrator or draw.io.
