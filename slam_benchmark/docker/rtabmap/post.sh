#!/bin/bash
# After RTAB-Map has stopped: export the FULL optimised graph from its database.
#
# /rtabmap/mapPath, recorded live, carries getLocalOptimizedPoses() -- the local
# window around the current node (the "local map=6" in the stats line), not the
# map. A run with 129 map updates and WM=83 yielded a seven-pose trajectory.
# The database has every node with its stamp, optimised, and RTAB-Map ships the
# exporter. Format 10 is TUM in the ROS frame; --poses_camera is the camera
# frame, which is the sensor frame this container was given (contract rule 1).
set -euo pipefail
set +u; source /opt/ros/humble/setup.bash; set -u
OUT=${OUT:-/out}
DB="$OUT/rtabmap.db"
[[ -s "$DB" ]] || { echo "post: no database at $DB, nothing to export" >&2; exit 0; }

cd "$OUT"
rtabmap-export --poses_camera --poses_format 10 --output_dir "$OUT" --output graph "$DB" \
    > "$OUT/export.log" 2>&1 || { echo "post: rtabmap-export failed, see export.log" >&2; exit 0; }

EXPORTED=$(ls "$OUT"/graph_camera_poses*.txt 2>/dev/null | head -1 || true)
[[ -n "$EXPORTED" && -s "$EXPORTED" ]] || { echo "post: export produced no poses file" >&2; exit 0; }

# The live capture becomes a labelled sibling; the database export is the
# trajectory. Both are kept, and timing.json says which is which.
[[ -f "$OUT/trajectory.tum" ]] && mv "$OUT/trajectory.tum" "$OUT/trajectory_mappath.tum"
{ echo "# timestamp tx ty tz qx qy qz qw"; grep -v '^#' "$EXPORTED"; } > "$OUT/trajectory.tum"
N=$(grep -vc '^#' "$OUT/trajectory.tum")
python3 - "$OUT/timing.json" "$N" <<'PY'
import json, sys
p, n = sys.argv[1], int(sys.argv[2])
try:
    d = json.load(open(p))
except Exception:
    d = {}
d.update({"frames": n, "trajectory_source": "database_export",
          "mappath_poses": d.get("frames")})
json.dump(d, open(p, "w"), indent=2)
PY
echo "post: $N optimised poses exported from the database -> trajectory.tum"
