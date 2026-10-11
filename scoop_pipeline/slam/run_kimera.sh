#!/bin/bash
# Kimera-VIO (MIT-SPARK/Kimera-VIO, ros2/base, built in Hydra's workspace) on a
# datasets/euroc.py export, every robot on its own, loop closure detection on.
# Run inside Hydra's container (hydra_ros/docker, PROFILE=dev), with the data
# and this repo mounted at the same paths as outside.
#
#   scoop_pipeline/slam/run_kimera.sh <euroc export dir> [robots "mobile_1 mobile_2"]
#
# Writes <export>/<robot>/out/: traj_vio.csv (Kimera-VIO's odometry),
# traj_pgo.csv (with its loop closures), output_lcd_result.csv (every loop
# closure query: is it one, the two keyframe stamps and their relative pose),
# kimera.log.
set -e
OUT=$(realpath "$1"); ROBOTS=(${2:-mobile_1 mobile_2})
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
  echo "$R: Kimera-VIO on $(($(wc -l < $D/mav0/cam0/data.csv) - 1)) stereo pairs ($O/kimera.log)"
  T0=$SECONDS
  set +e
  "$BIN" --dataset_type=0 --dataset_path="$D" --params_folder_path="$P" \
      --initial_k=50 --final_k=100000000 --use_lcd=true --vocabulary_path="$VOC" \
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
  echo "$R: done in $((SECONDS - T0)) s: $n_vio odometry poses, $n_lc loop closures"
done
