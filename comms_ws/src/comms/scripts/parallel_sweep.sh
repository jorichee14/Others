#!/usr/bin/env bash
# parallel_sweep.sh -- measure the knee in parallel TCP streams.
#
#   ./parallel_sweep.sh -s 192.168.51.11 -p 5201
#
# Run once per adapter, stationary and close to the AP, so the channel is not
# the limit and what you see is the TCP window/aggregation behaviour. Re-run
# after any PHY change (HE off, 20 -> 80 MHz), because the knee moves.
#
# The point is to replace an inherited default with a measured one. Output
# goes to a file you can cite: "throughput saturates at N parallel streams on
# this hardware (X Mbit/s at 1, Y at N, no gain to 8); N was used throughout."

set -uo pipefail

SERVER="192.168.51.11"
PORT=5201
DURATION=10
OMIT=2
REPEATS=2                      # repeat so a single noisy run cannot set the knee
STREAMS=(1 2 4 6 8)
REVERSE=""
OUT="parallel_sweep_$(date +%Y%m%d_%H%M%S).txt"

usage() {
    cat <<EOF
Usage: $0 [options]
  -s SERVER    iperf3 server            (default: ${SERVER})
  -p PORT      iperf3 port              (default: ${PORT})
  -t SECONDS   per-test duration        (default: ${DURATION})
  -O SECONDS   omitted warm-up          (default: ${OMIT})
  -n REPEATS   repeats per stream count (default: ${REPEATS})
  -R           downlink (iperf3 -R); the knee can differ per direction
  -o FILE      output file              (default: timestamped)
  -h           this help
EOF
    exit "${1:-0}"
}

while getopts "s:p:t:O:n:Ro:h" opt; do
    case "${opt}" in
        s) SERVER="${OPTARG}" ;;
        p) PORT="${OPTARG}" ;;
        t) DURATION="${OPTARG}" ;;
        O) OMIT="${OPTARG}" ;;
        n) REPEATS="${OPTARG}" ;;
        R) REVERSE="-R" ;;
        o) OUT="${OPTARG}" ;;
        h) usage 0 ;;
        *) usage 1 ;;
    esac
done

command -v iperf3 >/dev/null || { echo "error: iperf3 not on PATH" >&2; exit 1; }

DIR_LABEL=$([[ -n "${REVERSE}" ]] && echo downlink || echo uplink)

{
    echo "# parallel stream sweep"
    echo "# $(date -Iseconds)  host=$(hostname)  direction=${DIR_LABEL}"
    echo "# server=${SERVER}:${PORT}  -t ${DURATION} -O ${OMIT}  repeats=${REPEATS}"
    echo "# $(iperf3 --version 2>&1 | head -1)"
    echo "#"
    printf "%-4s %-8s %s\n" "-P" "run" "Mbit/s"
} | tee "${OUT}"

for p in "${STREAMS[@]}"; do
    for r in $(seq 1 "${REPEATS}"); do
        # Receiver-side goodput is the honest number: it is what actually
        # arrived, not what the sender pushed into its own queue.
        mbps=$(iperf3 -c "${SERVER}" -p "${PORT}" -t "${DURATION}" \
                      -O "${OMIT}" -P "${p}" ${REVERSE} -J 2>/dev/null \
               | python3 -c '
import json,sys
try:
    d = json.load(sys.stdin)
    e = d.get("end", {})
    s = e.get("sum_received") or e.get("sum_sent") or {}
    print(f"{s.get(\"bits_per_second\", 0)/1e6:.1f}")
except Exception:
    print("FAILED")
')
        printf "%-4s %-8s %s\n" "${p}" "${r}" "${mbps}" | tee -a "${OUT}"
        sleep 2          # let the link settle between tests
    done
done

echo | tee -a "${OUT}"
echo "# knee = lowest -P whose rate is within noise of the maximum." | tee -a "${OUT}"
echo "written to ${OUT}"
