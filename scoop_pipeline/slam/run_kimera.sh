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
#
# Writes <export>/<robot>/out/: traj_vio.csv (Kimera-VIO's odometry),
# traj_pgo.csv (with its loop closures), output_lcd_result.csv (every loop
# closure query: is it one, the two keyframe stamps and their relative pose),
# kimera.log.
set -e
OUT=$(realpath "$1"); ROBOTS=(${2:-mobile_1 mobile_2}); INITIAL_K=${INITIAL_K:-50}
WS=${HYDRA_WS:-/root/hydra_ws}
BIN=$WS/build/kimera_vio/stereoVIOEuroc
VOC=$WS/src/kimera_vio/vocabulary/ORBvoc.yml
[ -x "$BIN" ] || { echo "no $BIN: colcon build --packages-select kimera_vio first"; exit 1; }
[ -f "$VOC" ] || { echo "no $VOC: the kimera_vio build downloads it (or unzip ORBvoc.zip there)"; exit 1; }
source /opt/ros/$ROS_DISTRO/setup.bash
source $WS/install/setup.bash
for R in "${ROBOTS[@]}"; do
  D=$OUT/$R; P=$D/params; O=$D/out
  [ -f "$D/mav0/cam0/data.csv" ] || { echo "no $D/mav0: run datasets/euroc.py first"; exit 1; }
  [ -e "$O" ] && { echo "$O exists: remove it to run $R again"; exit 1; }
  mkdir -p "$O"
  # the last stereo pair with IMU after it: Kimera-VIO waits for IMU past a frame forever (an
  # IMU moved back onto the images' clock, euroc.py --imu-delay, ends before the last frames)
  t_imu=$(awk -F, 'END {print $1}' "$D/mav0/imu0/data.csv")
  FINAL_K=$(awk -F, -v t="$t_imu" '!/^#/ && $1 + 50000000 < t {n++} END {print n - 1}' "$D/mav0/cam0/data.csv")
  echo "$R: Kimera-VIO on $(($(wc -l < $D/mav0/cam0/data.csv) - 1)) stereo pairs from $INITIAL_K to $FINAL_K (the last with IMU after it), backend $(grep -m1 "^backend_type" $P/PipelineParams.yaml), autoInitialize $(grep -m1 "^autoInitialize" $P/BackendParams.yaml | awk '{print $2}') ($O/kimera.log)"
  T0=$SECONDS
  set +e
  "$BIN" --dataset_type=0 --dataset_path="$D" --params_folder_path="$P" \
      --initial_k=$INITIAL_K --final_k=$FINAL_K --use_lcd=true --vocabulary_path="$VOC" \
      --flagfile="$P/flags/stereoVIOEuroc.flags" --flagfile="$P/flags/Mesher.flags" \
      --flagfile="$P/flags/VioBackend.flags" --flagfile="$P/flags/RegularVioBackend.flags" \
      --flagfile="$P/flags/Visualizer3D.flags" --output_path="$O/" \
      --logtostderr=1 --colorlogtostderr=0 --v=0 > "$O/kimera.log" 2>&1
  rc=$?
  set -e
  chown -R --reference="$D" "$O" 2>/dev/null || true
  if [ $rc -ne 0 ]; then
    echo "$R: Kimera-VIO exited with $rc after $((SECONDS - T0)) s; the end of its log:"; tail -15 "$O/kimera.log"; exit 1
  fi
  n_vio=$(($(wc -l < "$O/traj_vio.csv" 2>/dev/null || echo 1) - 1))
  n_lc=$(awk -F, 'NR > 1 && $4 == 1' "$O/output_lcd_result.csv" 2>/dev/null | wc -l)
  # seconds of the data the odometry covers (its last pose after the first stereo pair)
  t0=$(awk -F, 'NR == 2 {print $1}' "$D/mav0/cam0/data.csv"); t1=$(awk -F, 'END {print $1}' "$D/mav0/cam0/data.csv")
  tv=$(awk -F, 'END {print $1}' "$O/traj_vio.csv" 2>/dev/null)
  span=$(awk -v a="$t0" -v b="${tv:-$t0}" -v c="$t1" 'BEGIN {printf "%.1f of %.1f s", (b - a) * 1e-9, (c - a) * 1e-9}')
  # the backend stops on a failed optimization and Kimera-VIO still exits with 0
  if grep -q "Variable has type" "$O/kimera.log"; then
    echo "$R: Kimera-VIO's backend failed after $n_vio odometry poses, $span of the data:"
    grep -m1 -B1 "Variable has type" "$O/kimera.log"; exit 1
  fi
  echo "$R: done in $((SECONDS - T0)) s: $n_vio odometry poses over $span of the data, $n_lc loop closures"
done
