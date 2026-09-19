#!/bin/bash
# MASt3R-SLAM: the learned backbone, and the one entry that is not a ROS node.
#
# So it replaces the shared entrypoint rather than plugging into it. The shape is
# the same and the contract is identical -- /out/trajectory.tum on the BAG's
# clock, /out/map.ply in the same world frame, /out/timing.json -- but the middle
# step is an offline run over a folder instead of a replay.
set -euo pipefail
source /opt/ros/humble/setup.bash

OUT=${OUT:-/out}
mkdir -p "$OUT"
: "${SLAM_COLOR_TOPIC:?}" "${SLAM_COLOR_INFO_TOPIC:?}"
T0=$(date +%s.%N)

# 1. Unroll the bag. The stamps come out beside the frames; step 3 puts them back.
python3 /opt/slambench/bag_to_frames.py \
    --bag /bag --topic "$SLAM_COLOR_TOPIC" \
    --info-topic "$SLAM_COLOR_INFO_TOPIC" --out /work/frames

# 2. Track. --calib feeds the bag's own rectified intrinsics, which is what
#    `use_calib: true` in the method config means; without it the run is the
#    uncalibrated experiment, which is a different row.
cd /opt/MASt3R-SLAM
python3 main.py --dataset /work/frames --config config/base.yaml \
    --calib /work/frames/calib.yaml --no-viz --save-as run

# 3. Back onto the bag clock. MASt3R's folder loader numbered the frames
#    index/30.0; restamp_tum.py inverts that exactly or refuses.
python3 /opt/slambench/restamp_tum.py \
    --traj logs/run/frames.txt --stamps /work/frames/stamps.txt \
    --out "$OUT/trajectory.tum" --fps 30.0

[[ -f logs/run/frames.ply ]] && cp logs/run/frames.ply "$OUT/map.ply"

python3 - "$OUT" "$T0" <<'PY'
import json, resource, subprocess, sys, time
out, t0 = sys.argv[1], float(sys.argv[2])
n = sum(1 for ln in open(f"{out}/trajectory.tum") if ln.strip() and not ln.startswith("#"))
ru = resource.getrusage(resource.RUSAGE_CHILDREN)
json.dump({"frames": n, "wall_s": time.time() - t0,
           "cpu_s": ru.ru_utime + ru.ru_stime,
           "peak_rss_mb": ru.ru_maxrss / 1024.0,
           # KEYFRAMES, not every frame: MASt3R-SLAM saves its keyframe poses
           # only. The evaluator associates by stamp, so a sparse trajectory is
           # legal -- but it is a smaller sample than the RGB-D rows and the
           # count belongs beside the ATE, not buried.
           "poses_are": "keyframes"},
          open(f"{out}/timing.json", "w"), indent=2)
PY
test -s "$OUT/trajectory.tum" || { echo "no trajectory written" >&2; exit 3; }
wc -l "$OUT/trajectory.tum"
