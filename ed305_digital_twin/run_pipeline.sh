#!/usr/bin/env bash
# The whole ED305 room pipeline on one recording session, end to end:
#   1. cameras in the room frame (once per calibration)   make_room_cameras.py
#   2. recordings on one clock (once per session)          motion_sync.py
#   3. tracking + trust + combined video                    room_track.py --video all
#   4. paper figures                                        make_figures.py
#
# Usage (from ed305_digital_twin/, conda env active):
#   REC=~/Downloads/yolo_20261105_10 ./run_pipeline.sh
#   REC=~/rec START=60 END=180 OUT=out/run1 REID=yolo26s-reid.onnx ./run_pipeline.sh
#
# Settings (environment variables, defaults in brackets):
#   REC          folder with one cam_N.mkv/.mp4 per camera                         [required]
#   CALIB        cameras in the room frame                     [configs/cameras_room.yaml]
#   EXTRINSICS   LiDAR-map camera poses, used only when CALIB does not exist yet
#   INTRINSICS   lens calibration folder                            [configs/intrinsics]
#   FLOOR_MAP    LiDAR map (.ply/.pcd) to fit the floor in, used with EXTRINSICS
#   MODEL        person detector                   [runs/detect/ed305_v1/weights/best.pt]
#   REID         appearance model, empty = none                       [yolo26s-reid.onnx]
#   DEVICE       GPU id or cpu                                                         [0]
#   START, END   window on the common clock in s, empty = all                     [empty]
#   RATE         time steps per second                                                [10]
#   SENSORS      sensor models / door zone YAML for room_track, empty = none        [empty]
#   EXTRA        extra measurement CSVs (radar, robots), space-separated            [empty]
#   OUT          output folder                                                [out/run]
#   RESYNC=1     run motion_sync again even if times files exist
set -euo pipefail
cd "$(dirname "$0")"

: "${REC:?set REC to the folder with the cam_N recordings}"
CALIB=${CALIB:-configs/cameras_room.yaml}
INTRINSICS=${INTRINSICS:-configs/intrinsics}
MODEL=${MODEL:-runs/detect/ed305_v1/weights/best.pt}
REID=${REID-yolo26s-reid.onnx}
DEVICE=${DEVICE:-0}
RATE=${RATE:-10}
OUT=${OUT:-out/run}
step() { printf '\n== %s\n' "$*"; }

step "1/4 cameras in the room frame: $CALIB"
if [[ -f "$CALIB" ]]; then
  echo "exists, kept (delete it to rebuild from EXTRINSICS)"
else
  : "${EXTRINSICS:?$CALIB does not exist: set EXTRINSICS (and FLOOR_MAP) to build it}"
  : "${FLOOR_MAP:?set FLOOR_MAP to the LiDAR map (.ply/.pcd) to fit the floor in}"
  python make_room_cameras.py --extrinsics "$EXTRINSICS" --intrinsics "$INTRINSICS" \
    --floor-map "$FLOOR_MAP" --out "$CALIB"
fi

step "2/4 recordings on one clock: $REC"
shopt -s nullglob
vids=("$REC"/cam_*.mkv "$REC"/cam_*.mp4)
missing=0
for v in "${vids[@]}"; do [[ -f "${v%.*}_times.csv" ]] || missing=1; done
if [[ ${RESYNC:-0} == 1 || $missing == 1 ]]; then
  python motion_sync.py "$REC"
else
  echo "every recording has a times file, kept (RESYNC=1 to redo)"
fi

step "3/4 tracking, trust, combined video -> $OUT"
args=(--rec "$REC" --calib "$CALIB" --model "$MODEL" --device "$DEVICE" --rate "$RATE" --video all --out "$OUT")
[[ -n "$REID" ]] && args+=(--reid "$REID")
[[ -n "${START:-}" ]] && args+=(--start "$START")
[[ -n "${END:-}" ]] && args+=(--end "$END")
[[ -n "${SENSORS:-}" ]] && args+=(--sensors "$SENSORS")
# shellcheck disable=SC2206
[[ -n "${EXTRA:-}" ]] && args+=(--measurements $EXTRA)
python room_track.py "${args[@]}"

step "4/4 figures -> $OUT/figures"
python make_figures.py "$OUT" --calib "$CALIB"

printf '\nDone. Open %s/all_ids.mp4 and %s/figures/. Trust per person: %s/persons_state.csv\n' "$OUT" "$OUT" "$OUT"
