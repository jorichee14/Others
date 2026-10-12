#!/bin/bash
# Kimera-VIO (MIT-SPARK/Kimera-VIO, ros2/base, built in Hydra's workspace) on a
# datasets/euroc.py export, every robot on its own, loop closure detection on.
# Run inside Hydra's container (hydra_ros/docker, PROFILE=dev), with the data
# and this repo mounted at the same paths as outside.
#
#   scoop_pipeline/slam/run_kimera.sh <euroc export dir> [robots "mobile_1 mobile_2"]
#
# INITIAL_K=<k> (environment): start at stereo pair k (default 50, as Kimera-VIO's
#   own script), e.g. after a long stationary start.
# STRETCHES=1 (environment): one run per stretch without a dropout
#   (slam/kimera_stretches.py: no IMU, camera or ground-truth gap over GAP=0.5 s,
#   at least MIN_S=10 s), each starting from the ground truth at its first frame
#   (euroc.py's autoInitialize 0): for a robot whose sensor dropped out, as
#   mobile_2's RealSense on 20260924. A stretch that fails does not stop the rest.
#
# Writes <export>/<robot>/out/ (STRETCHES=1: out/stretch_<n>/ each, and
# out/stretches.txt "n k0 k1 t0 t1"): traj_vio.csv (Kimera-VIO's odometry),
# traj_pgo.csv (with its loop closures), output_lcd_result.csv (every loop
# closure query: is it one, the two keyframe stamps and their relative pose),
# kimera.log. analysis/slam/kimera_eval.py evaluates either.
set -e
OUT=$(realpath "$1"); ROBOTS=(${2:-mobile_1 mobile_2}); INITIAL_K=${INITIAL_K:-50}
HERE=$(dirname "$(realpath "$0")")
WS=${HYDRA_WS:-/root/hydra_ws}
BIN=$WS/build/kimera_vio/stereoVIOEuroc
VOC=$WS/src/kimera_vio/vocabulary/ORBvoc.yml
[ -x "$BIN" ] || { echo "no $BIN: colcon build --packages-select kimera_vio first"; exit 1; }
[ -f "$VOC" ] || { echo "no $VOC: the kimera_vio build downloads it (or unzip ORBvoc.zip there)"; exit 1; }
source /opt/ros/$ROS_DISTRO/setup.bash
source $WS/install/setup.bash

# run_one <robot dir> <out dir> <k0> <k1> <label>: one Kimera-VIO run; 0 when it got through
run_one() {
  local D=$1 O=$2 K0=$3 K1=$4 L=$5 P=$1/params
  mkdir -p "$O"
  local T0=$SECONDS rc
  set +e
  "$BIN" --dataset_type=0 --dataset_path="$D" --params_folder_path="$P" \
      --initial_k=$K0 --final_k=$K1 --use_lcd=true --vocabulary_path="$VOC" \
      --flagfile="$P/flags/stereoVIOEuroc.flags" --flagfile="$P/flags/Mesher.flags" \
      --flagfile="$P/flags/VioBackend.flags" --flagfile="$P/flags/RegularVioBackend.flags" \
      --flagfile="$P/flags/Visualizer3D.flags" --output_path="$O/" \
      --logtostderr=1 --colorlogtostderr=0 --v=0 > "$O/kimera.log" 2>&1
  rc=$?
  set -e
  chown -R --reference="$D" "$O" 2>/dev/null || true
  if [ $rc -ne 0 ]; then
    echo "$L: Kimera-VIO exited with $rc after $((SECONDS - T0)) s; the end of its log:"; tail -15 "$O/kimera.log"
    return 1
  fi
  local n_vio n_lc t0 t1 tv span
  n_vio=$(($(wc -l < "$O/traj_vio.csv" 2>/dev/null || echo 1) - 1))
  n_lc=$(awk -F, 'NR > 1 && $4 == 1' "$O/output_lcd_result.csv" 2>/dev/null | wc -l)
  # seconds the odometry covers, after the first stereo pair of the data
  t0=$(awk -F, 'NR == 2 {print $1}' "$D/mav0/cam0/data.csv"); t1=$(awk -F, 'END {print $1}' "$D/mav0/cam0/data.csv")
  tv=$(awk -F, 'END {print $1}' "$O/traj_vio.csv" 2>/dev/null)
  span=$(awk -v a="$t0" -v b="${tv:-$t0}" -v c="$t1" 'BEGIN {printf "up to %.1f of %.1f s", (b - a) * 1e-9, (c - a) * 1e-9}')
  # the backend stops on a failed optimization and Kimera-VIO still exits with 0
  if grep -q "Variable has type" "$O/kimera.log"; then
    echo "$L: Kimera-VIO's backend failed after $n_vio odometry poses, $span of the data:"
    grep -m1 -B1 "Variable has type" "$O/kimera.log"
    return 1
  fi
  echo "$L: done in $((SECONDS - T0)) s: $n_vio odometry poses $span of the data, $n_lc loop closures"
}

for R in "${ROBOTS[@]}"; do
  D=$OUT/$R; P=$D/params; O=$D/out
  [ -f "$D/mav0/cam0/data.csv" ] || { echo "no $D/mav0: run datasets/euroc.py first"; exit 1; }
  [ -e "$O" ] && { echo "$O exists: remove it to run $R again"; exit 1; }
  mkdir -p "$O"
  setup="backend $(grep -m1 "^backend_type" $P/PipelineParams.yaml), autoInitialize $(grep -m1 "^autoInitialize" $P/BackendParams.yaml | awk '{print $2}')"
  if [ -n "${STRETCHES:-}" ]; then
    python3 "$HERE/kimera_stretches.py" "$D" --gap "${GAP:-0.5}" --min "${MIN_S:-10}" --initial-k "$INITIAL_K" \
        | awk '{print NR, $0}' > "$O/stretches.txt"
    echo "$R: Kimera-VIO on $(wc -l < "$O/stretches.txt") stretches without a dropout, $setup:"
    awk '{printf "  stretch %d: pairs %d-%d, %.1f-%.1f s\n", $1, $2, $3 - 1, $4, $5}' "$O/stretches.txt"
    ok=0
    while read -r n k0 k1 a b; do
      run_one "$D" "$O/stretch_$n" "$k0" "$k1" "$R stretch $n (${a}-${b} s)" < /dev/null && ok=$((ok + 1)) || true
    done < "$O/stretches.txt"
    echo "$R: $ok of $(wc -l < "$O/stretches.txt") stretches through"
    continue
  fi
  # the last stereo pair with IMU after it: Kimera-VIO waits for IMU past a frame forever (an
  # IMU moved back onto the images' clock, euroc.py --imu-delay, ends before the last frames)
  t_imu=$(awk -F, 'END {print $1}' "$D/mav0/imu0/data.csv")
  FINAL_K=$(awk -F, -v t="$t_imu" '!/^#/ && $1 + 50000000 < t {n++} END {print n - 1}' "$D/mav0/cam0/data.csv")
  echo "$R: Kimera-VIO on $(($(wc -l < $D/mav0/cam0/data.csv) - 1)) stereo pairs from $INITIAL_K to $FINAL_K (the last with IMU after it), $setup ($O/kimera.log)"
  run_one "$D" "$O" "$INITIAL_K" "$FINAL_K" "$R" || exit 1
done
