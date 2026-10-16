#!/usr/bin/env bash
# Koide's direct_visual_lidar_calibration (docker) on the pause bags of
# processing/dvlc_pauses.py, its output owned by you afterwards (no sudo).
#
#   processing/run_dvlc.sh <step> <pauses_dir> [preprocessed_dir]
#
#   step              preprocess | guess | calibrate | viewer | all
#                     (guess = initial_guess_manual; all = the four in order)
#   pauses_dir        dvlc_pauses.py's output: pause_00/, pause_01/, ...
#   preprocessed_dir  default: <pauses_dir>_preprocessed
#
# Topics (env): IMAGE_TOPIC (/mobile_2/color/image_raw), POINTS_TOPIC
# (/mobile_2/ouster/points), CAMERA_INFO_TOPIC (/mobile_2/color/camera_info).
# Image (env): DVLC_IMAGE (dvlc:humble-mcap). Only the pause_* folders are
# mounted (read-only), so pauses.yaml never reaches preprocess.
# The result, calibrate's calib.json, is printed when it is there.
set -euo pipefail

if [ $# -lt 2 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    sed -n '2,18p' "$0"; exit 1
fi
STEP=$1
BAG=$(realpath "$2")
PRE=$(realpath -m "${3:-${BAG%/}_preprocessed}")
IMAGE=${DVLC_IMAGE:-dvlc:humble-mcap}
IMAGE_TOPIC=${IMAGE_TOPIC:-/mobile_2/color/image_raw}
POINTS_TOPIC=${POINTS_TOPIC:-/mobile_2/ouster/points}
CAMERA_INFO_TOPIC=${CAMERA_INFO_TOPIC:-/mobile_2/color/camera_info}

case "$STEP" in
    preprocess|guess|calibrate|viewer|all) ;;
    *) echo "step must be preprocess, guess, calibrate, viewer or all (got $STEP)"; exit 1 ;;
esac

MOUNTS=()
for d in "$BAG"/pause_*/; do
    [ -f "$d/metadata.yaml" ] || continue
    n=$(basename "$d")
    MOUNTS+=(-v "$BAG/$n":/tmp/input_bags/"$n":ro)
done
[ ${#MOUNTS[@]} -gt 0 ] || { echo "no pause_*/metadata.yaml in $BAG: run dvlc_pauses.py first"; exit 1; }
mkdir -p "$PRE"         # created by you, so docker does not create it as root

echo "pauses:       $BAG ($(( ${#MOUNTS[@]} / 2 )) bags)"
echo "preprocessed: $PRE"
xhost +local:root >/dev/null 2>&1 || true  # let the container open its windows

PKG=direct_visual_lidar_calibration
OWNER="$(id -u):$(id -g)"
TTY=(); [ -t 0 ] && TTY=(-it)

dvlc() {
    # the image keeps ROS under /root, so it runs as root; the output is
    # handed back to you when it exits -- also on failure or Ctrl+C
    docker run "${TTY[@]}" --rm --net host --gpus all \
        -e DISPLAY="${DISPLAY:-:0}" -v "$HOME/.Xauthority":/root/.Xauthority \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        "${MOUNTS[@]}" -v "$PRE":/tmp/preprocessed \
        "$IMAGE" \
        bash -c "trap 'chown -R $OWNER /tmp/preprocessed' EXIT; $*"
}

run() {
    echo; echo "== $1"
    case "$1" in
        preprocess) dvlc ros2 run $PKG preprocess /tmp/input_bags /tmp/preprocessed -v \
                        --image_topic "$IMAGE_TOPIC" --points_topic "$POINTS_TOPIC" \
                        --camera_info_topic "$CAMERA_INFO_TOPIC" ;;
        guess)      dvlc ros2 run $PKG initial_guess_manual /tmp/preprocessed ;;
        calibrate)  dvlc ros2 run $PKG calibrate /tmp/preprocessed ;;
        viewer)     dvlc ros2 run $PKG viewer /tmp/preprocessed ;;
    esac
}

if [ "$STEP" = all ]; then
    for s in preprocess guess calibrate viewer; do run $s; done
else
    run "$STEP"
fi

echo
if [ -f "$PRE/calib.json" ]; then
    echo "result: $PRE/calib.json"
    python3 -c "import json,sys; r=json.load(open(sys.argv[1])).get('results',{}); \
[print(f'  {k}: {v}') for k,v in r.items()]" "$PRE/calib.json" || true
fi
