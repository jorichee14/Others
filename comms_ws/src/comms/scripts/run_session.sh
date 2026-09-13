#!/usr/bin/env bash
# run_session.sh -- one traverse of a route: stack + bag, started and stopped together.
#
#   ./run_session.sh -m capacity -r mobile_1 -l floor3_eastwing
#   ./run_session.sh -m deadline -r mobile_1 -l floor3_eastwing -o busy
#
# Runs until you press Ctrl-C, because a traverse takes as long as it takes.
# On Ctrl-C the bag is closed BEFORE the publishers are killed, so the last
# samples are written rather than lost.
#
# Two traverses per route, back to back so the environment is comparable:
#
#   -m capacity   continuous iperf at 1 Hz + passive monitor. Saturates the
#                 link, so this is throughput and loaded RTT everywhere.
#   -m deadline   bursts + ping on an idle link. Deadline compliance and
#                 unloaded latency everywhere.
#
# They cannot share a traverse: capacity fills the AP's queue and deadline
# needs it empty, and a continuously-running stack has no gap to settle in.
#
# Segmentation is offline from LiDAR pose -- nothing to type while driving.

set -uo pipefail

MODE="capacity"
ROBOT="mobile_1"
LABEL="session"
OCCUPANCY="unknown"
NOTE=""
OUT_ROOT="${HOME}/comms_data"
SERVER="192.168.51.11"
PARALLEL=2                   # from parallel_sweep.sh -- fix for the campaign
IVAL=1.0                     # capacity sample rate: 1.0 => 1 m/sample at 1 m/s
BURST_BYTES=40960
DEADLINE_MS=50.0
REVERSE="false"

usage() {
    awk 'NR>1 { if ($0 !~ /^#/) exit; sub(/^# ?/, ""); print }' "$0"
    cat <<EOF

Options:
  -m MODE        capacity | deadline            (default: ${MODE})
  -r ROBOT       mobile_1 | mobile_2            (default: ${ROBOT})
  -l LABEL       route label for the bag name   (default: ${LABEL})
  -o OCCUPANCY   empty | few | busy | unknown   (default: ${OCCUPANCY})
  -n NOTE        free text, quoted
  -s SERVER      iperf3 / echo host             (default: ${SERVER})
  -d OUT_ROOT    output root                    (default: ${OUT_ROOT})
  -R             capacity mode: downlink instead of uplink
  -h             this help
EOF
    exit "${1:-0}"
}

while getopts "m:r:l:o:n:s:d:Rh" opt; do
    case "${opt}" in
        m) MODE="${OPTARG}" ;;
        r) ROBOT="${OPTARG}" ;;
        l) LABEL="${OPTARG}" ;;
        o) OCCUPANCY="${OPTARG}" ;;
        n) NOTE="${OPTARG}" ;;
        s) SERVER="${OPTARG}" ;;
        d) OUT_ROOT="${OPTARG}" ;;
        R) REVERSE="true" ;;
        h) usage 0 ;;
        *) usage 1 ;;
    esac
done

case "${MODE}" in
    capacity|deadline) ;;
    *) echo "error: mode must be 'capacity' or 'deadline'" >&2; exit 1 ;;
esac

# Ports differ per robot so two robots never share one iperf3 server: a
# single server serves one test at a time, and the second client queueing
# behind the first would look like contention without being it.
case "${ROBOT}" in
    mobile_1) IFACE="wlx8876b9eae0ff"; PORT=5201; BURST_PORT=5600; CSI="/mobile_1/csi" ;;
    mobile_2) IFACE="wlx8876b9eae101"; PORT=5203; BURST_PORT=5601; CSI="/mobile_2/csi" ;;
    *) echo "error: unknown robot '${ROBOT}'" >&2; exit 1 ;;
esac

STAMP="$(date +%Y%m%d_%H%M%S)"
DIR="${OUT_ROOT}/${LABEL}/${ROBOT}_${MODE}_${STAMP}"
mkdir -p "${DIR}" || { echo "error: cannot create ${DIR}" >&2; exit 1; }

QOS="$(ros2 pkg prefix comms 2>/dev/null)/share/comms/config/rosbag_qos_override.yaml"
[[ -f "${QOS}" ]] || { echo "error: QoS override not found at ${QOS}" >&2; exit 1; }

# Both modes record the same topic list. Topics belonging to the inactive
# mode simply carry nothing, which keeps every bag the same shape and makes
# the analysis code mode-agnostic.
TOPICS=(
    "/${ROBOT}/wifi/status"
    "/${ROBOT}/wifi/iperf_up"
    "/${ROBOT}/wifi/iperf_down"
    "/${ROBOT}/wifi/ping"
    "/${ROBOT}/wifi/burst"
    "/${ROBOT}/ntp/client/status"
    "${CSI}"
    "/csi_publisher/status"
    "/tf" "/tf_static" "/odom" "/scan"
)

LAUNCH_PID=""
BAG_PID=""
STOPPED=0

stop_all() {
    [[ "${STOPPED}" -eq 1 ]] && return
    STOPPED=1
    echo
    echo "stopping..."
    # Bag first: the publishers must outlive it or the tail of the session
    # is lost.
    if [[ -n "${BAG_PID}" ]]; then
        kill -INT "${BAG_PID}" 2>/dev/null
        wait "${BAG_PID}" 2>/dev/null
    fi
    if [[ -n "${LAUNCH_PID}" ]]; then
        kill -INT "${LAUNCH_PID}" 2>/dev/null
        wait "${LAUNCH_PID}" 2>/dev/null
    fi
}
trap stop_all INT TERM

echo "=============================================="
echo " ${MODE} traverse | ${ROBOT} | ${IFACE}"
echo " server ${SERVER}  route '${LABEL}'"
if [[ "${MODE}" == "capacity" ]]; then
    echo " iperf continuous, -P ${PARALLEL}, ${IVAL}s samples, reverse=${REVERSE}"
    echo " NOTE: link is saturated for the whole traverse"
else
    echo " bursts ${BURST_BYTES}B, deadline ${DEADLINE_MS}ms, + ping"
    echo " NOTE: needs burst_echo_server.py on ${SERVER}:${BURST_PORT}"
fi
echo " output ${DIR}"
echo " drive the route, then Ctrl-C to finish"
echo "=============================================="

ros2 launch comms session.launch.py \
    mode:="${MODE}" \
    server_address:="${SERVER}" server_port:="${PORT}" \
    burst_port:="${BURST_PORT}" \
    interface:="${IFACE}" namespace:="${ROBOT}" \
    parallel:="${PARALLEL}" continuous_interval_s:="${IVAL}" \
    reverse:="${REVERSE}" \
    burst_bytes:="${BURST_BYTES}" deadline_ms:="${DEADLINE_MS}" \
    run_ntp:=true \
    > "${DIR}/launch.log" 2>&1 &
LAUNCH_PID=$!

sleep 4                       # let the nodes come up before recording starts

if ! kill -0 "${LAUNCH_PID}" 2>/dev/null; then
    echo "error: launch died on startup -- see ${DIR}/launch.log" >&2
    tail -20 "${DIR}/launch.log" >&2
    exit 1
fi

ros2 bag record --qos-profile-overrides-path "${QOS}" \
    -o "${DIR}/bag" "${TOPICS[@]}" \
    > "${DIR}/bag.log" 2>&1 &
BAG_PID=$!

echo "recording (pid ${BAG_PID}) -- Ctrl-C when the traverse is done"
wait "${BAG_PID}" 2>/dev/null
stop_all

# --------------------------------------------------------------- metadata
# python rather than a heredoc: `iw dev link` contains tabs and newlines and
# a note can contain quotes, all of which break raw-pasted JSON.
export MODE ROBOT IFACE LABEL STAMP OCCUPANCY NOTE SERVER PORT BURST_PORT
export PARALLEL IVAL BURST_BYTES DEADLINE_MS REVERSE

python3 - "${DIR}/session.json" <<'PYEOF'
import json, os, subprocess, sys

def sh(cmd):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True,
                           text=True, timeout=10)
        return (r.stdout or r.stderr or "").strip() or "unavailable"
    except Exception:
        return "unavailable"

iface = os.environ["IFACE"]
mode = os.environ["MODE"]
meta = {
    "route_label":   os.environ["LABEL"],
    "mode":          mode,
    "robot":         os.environ["ROBOT"],
    "interface":     iface,
    "timestamp":     os.environ["STAMP"],
    "occupancy":     os.environ["OCCUPANCY"],
    "note":          os.environ["NOTE"],
    "server":        os.environ["SERVER"],
    "iperf_port":    int(os.environ["PORT"]),
    "burst_port":    int(os.environ["BURST_PORT"]),
    # Link state as it actually was. Months later this is the only record of
    # which PHY was negotiated and whether a shaper was left in place.
    "iw_link":       sh(f"iw dev {iface} link"),
    "qdisc":         sh(f"tc qdisc show dev {iface} | head -1"),
    "he_setting":    sh("grep -h rtw_he_enable /etc/modprobe.d/8852bu.conf"),
    "host":          sh("hostname"),
    "ros_domain_id": os.environ.get("ROS_DOMAIN_ID", "unset"),
}
if mode == "capacity":
    meta.update({
        "parallel":              int(os.environ["PARALLEL"]),
        "continuous_interval_s": float(os.environ["IVAL"]),
        "reverse":               os.environ["REVERSE"] == "true",
    })
else:
    meta.update({
        "burst_bytes": int(os.environ["BURST_BYTES"]),
        "deadline_ms": float(os.environ["DEADLINE_MS"]),
    })

with open(sys.argv[1], "w") as f:
    json.dump(meta, f, indent=2)
    f.write("\n")
PYEOF

echo
echo "done: ${DIR}"
ls -1 "${DIR}"
if [[ "${MODE}" == "capacity" ]]; then
    echo
    echo "next: ./run_session.sh -m deadline -r ${ROBOT} -l ${LABEL} -o ${OCCUPANCY}"
fi
