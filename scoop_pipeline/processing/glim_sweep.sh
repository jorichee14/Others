#!/usr/bin/env bash
# Run GLIM with several config folders on one mapping bag and score each
# trajectory against a reference, to choose GLIM's config by measurement.
#
#   processing/glim_sweep.sh <pipeline config> <reference traj> <glim config dir>...
#
#   e.g. processing/glim_sweep.sh map_stages/pipeline_config_mapping_20261015.json \
#            traj_lidar_ba_glim.txt configs/glim/variants/*/
#
# The bag is the config's dataset.bag; a bare reference name is in
# odometry/<machine>/. Each variant runs through processing/run_glim.sh into
# <folder of the bag>/glim_sweep/<variant>/ (kept: a variant already run is
# not run again -- delete its folder to redo it), then gets its loop closure
# on the boards (map_stages/loop_check.py), and all are compared with
# analysis/odom/compare_trajs.py --ref: ATE / RPE against the reference,
# next to the current GLIM run, 02 and BA. Report and plot in glim_sweep/.
#
# Image: $GLIM_IMAGE, default koide3/glim_ros2:humble_cuda12.2-mcap (it has
# the CPU modules as well, so every variant runs on the same build).
set -euo pipefail
if [ $# -lt 3 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    sed -n '2,19p' "$0"; exit 1
fi
HERE=$(cd "$(dirname "$0")/.." && pwd)
CFG=$(realpath "$1"); REF=$2; shift 2
export GLIM_IMAGE=${GLIM_IMAGE:-koide3/glim_ros2:humble_cuda12.2-mcap}

read -r BAG REFPATH < <(cd "$HERE/map_stages" && python3 - "$CFG" "$REF" <<'PY'
import os, sys
from pipeline_common import load_pipeline
P = load_pipeline(sys.argv[1])
print(os.path.expanduser(P.dataset["bag"]), P.outp(sys.argv[2]))
PY
)
[ -f "$REFPATH" ] || { echo "no reference trajectory at $REFPATH"; exit 1; }
OUT=$(dirname "$BAG")/glim_sweep
mkdir -p "$OUT"
echo "bag:       $BAG"
echo "reference: $REFPATH"
echo "image:     $GLIM_IMAGE"
echo "out:       $OUT"

ARGS=()
for V in "$@"; do
    V=$(realpath "$V"); NAME=$(basename "$V"); DUMP=$OUT/$NAME
    if [ -f "$DUMP/traj_lidar.txt" ]; then
        echo "== $NAME: already run ($DUMP)"
    else
        echo "== $NAME: running GLIM"
        rm -rf "$DUMP"
        "$HERE/processing/run_glim.sh" "$BAG" "$V" "$DUMP"
    fi
    [ -f "$DUMP/traj_lidar.txt" ] || { echo "   $NAME: no trajectory, skipped"; continue; }
    if [ ! -f "$DUMP/traj_lidar_loop_closure.json" ]; then
        (cd "$HERE" && python3 map_stages/loop_check.py "$CFG" --traj "$DUMP/traj_lidar.txt" \
            --out "$DUMP/traj_lidar_loop_closure.json" > "$DUMP/loop_check.log" 2>&1) \
            || echo "   $NAME: loop closure not measured (see $DUMP/loop_check.log)"
    fi
    ARGS+=(--traj "$NAME=$DUMP/traj_lidar.txt")
done

cd "$HERE"
python3 analysis/odom/compare_trajs.py "$CFG" "${ARGS[@]}" --traj "ref=$REFPATH" --ref ref \
    --out "$OUT/sweep.json" --plot "$OUT/sweep.png"
