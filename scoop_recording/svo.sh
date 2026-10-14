#!/bin/bash
# Start / stop the ZED .svo2 recording, named after the bag record.sh started.
#
#   ./svo.sh start     # after record.sh is running on this machine
#   ./svo.sh stop
#
# record.sh leaves the bag path in ./.last_run (run this from the same folder,
# normally Dataset/), so the .svo and the bag share one base name and land in the
# same folder. A name that already exists is refused: that means ./.last_run is
# from an earlier pass (record.sh was not started first) or a pass is being
# repeated, and overwriting an svo cannot be undone.
set -eo pipefail

LAST=${LAST_RUN:-./.last_run}

case "${1:?usage: svo.sh start|stop}" in
start)
    [ -f "$LAST" ] || { echo "no $LAST -- start record.sh first"; exit 1; }
    BAG=$(cat "$LAST")
    SVO="${BAG}.svo"
    if [ -e "$SVO" ] && [ "${FORCE:-0}" != "1" ]; then
        echo "$SVO exists. Start record.sh for this pass first (it makes a new name),"
        echo "or FORCE=1 to overwrite."
        exit 1
    fi
    echo "svo -> $SVO"
    ros2 service call /zed/zed_node/start_svo_rec zed_msgs/srv/StartSvoRec \
        "{bitrate: 0, compression_mode: 0, target_framerate: 0, input_transcode: false, svo_filename: '$SVO'}"
    ;;
stop)
    ros2 service call /zed/zed_node/stop_svo_rec std_srvs/srv/Trigger
    ;;
*)  echo "usage: svo.sh start|stop"; exit 1 ;;
esac
