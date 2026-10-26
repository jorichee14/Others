#!/bin/bash
# In the odometry container (run_odometry.sh starts it): the probe and the
# resource sampler first, then the method from /work/run.json, then the bag;
# everything stopped in reverse order and summarized, also when interrupted.
set -uo pipefail
set +u
source /opt/ros/humble/setup.bash
[ -f /ws/install/setup.bash ] && source /ws/install/setup.bash
set -u
W=/work; M=/opt/monitor; L=$W/logs
# large messages (the Ouster's 3 MB clouds) through shared memory, whole
export FASTRTPS_DEFAULT_PROFILES_FILE=${FASTRTPS_DEFAULT_PROFILES_FILE:-/opt/odometry/fastdds.xml}
mkdir -p "$L"

# run.json -> shell variables
eval "$(python3 - <<'EOF'
import json, shlex
r = json.load(open("/work/run.json"))
v = {"METHOD": r["method"], "LAUNCH": r["launch"], "PROC": r["proc_regex"],
     "OUT_TOPIC": r["out_topic"], "PLAY": " ".join(r["play_topics"]),
     "START_S": str(r.get("start_s") or 0), "DURATION_S": str(r.get("duration_s") or 0),
     "RELAY_IN": (r.get("relay") or {}).get("in", ""), "RELAY_OUT": (r.get("relay") or {}).get("out", ""),
     "RELAY_HOLD": str((r.get("relay") or {}).get("hold_s", 0))}
v["PROBE_ARGS"] = " ".join(["--in %s=%s" % kv for kv in r["in_topics"].items()]
                           + ["--out %s=%s" % (r["out_topic"], r["out_type"]), "--tum " + r["out_topic"]])
for k, x in v.items():
    print("%s=%s" % (k, shlex.quote(x)))
EOF
)"

PROBE=""; RES=""; SLAM=""; PLAY_PID=""; RELAY=""
stop_group() {                         # SIGINT to a process group, then insist after $2 s
    local pid=$1 t=0
    [ -n "$pid" ] || return 0
    kill -INT -- -"$pid" 2>/dev/null || kill -INT "$pid" 2>/dev/null || return 0
    while kill -0 "$pid" 2>/dev/null && [ "$t" -lt "$2" ]; do sleep 1; t=$((t + 1)); done
    kill -KILL -- -"$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
}
STATUS_METHOD=running; STATUS_PLAY=not_started
finish() {
    trap '' INT TERM
    stop_group "$PLAY_PID" 5
    stop_group "$SLAM" 20
    stop_group "$RELAY" 5
    stop_group "$PROBE" 10
    stop_group "$RES" 5
    python3 - "$STATUS_METHOD" "$STATUS_PLAY" <<'EOF'
import json, sys
json.dump({"method": sys.argv[1], "play": sys.argv[2]}, open("/work/status.json", "w"), indent=1)
EOF
    python3 "$M/summarize.py" "$W" || echo "summarize failed" >&2
    [ -n "${OWNER:-}" ] && chown -R "$OWNER" "$W" 2>/dev/null
    return 0
}
interrupted() { STATUS_PLAY=interrupted; touch "$W/INTERRUPTED"; finish; exit 130; }
trap interrupted INT TERM

echo "method : $METHOD"
echo "launch : $LAUNCH"
echo "play   : $PLAY (rate ${RATE:-1.0}, from ${START_S} s, ${DURATION_S} s)"
setsid python3 "$M/probe.py" --dir "$W" $PROBE_ARGS >"$L/probe.log" 2>&1 & PROBE=$!
setsid python3 "$M/resources.py" --dir "$W" --match "$PROC" >"$L/resources.log" 2>&1 & RES=$!
if [ -n "$RELAY_IN" ]; then
    echo "relay  : $RELAY_IN -> $RELAY_OUT, ${RELAY_HOLD} s later (the sweep's end)"
    setsid python3 /opt/odometry/sweep_relay.py --in "$RELAY_IN" --out "$RELAY_OUT" \
        --hold-s "$RELAY_HOLD" --rate "${RATE:-1.0}" >"$L/relay.log" 2>&1 & RELAY=$!
fi
setsid bash -c "$LAUNCH" >"$L/method.log" 2>&1 & SLAM=$!
sleep "${START_WAIT:-8}"
for p in PROBE RES SLAM $([ -n "$RELAY" ] && echo RELAY); do
    if ! kill -0 "${!p}" 2>/dev/null; then
        echo "$p exited at start:" >&2
        case $p in PROBE) tail -n 20 "$L/probe.log";; RES) tail -n 20 "$L/resources.log";;
                   SLAM) tail -n 30 "$L/method.log";; RELAY) tail -n 20 "$L/relay.log";; esac >&2
        STATUS_METHOD=exited_at_start; finish; exit 4
    fi
done

# --delay: the player's publishers exist a second before the first message, so
# the probe has subscribed to every input by then
ARGS=(--clock --rate "${RATE:-1.0}" --delay 1 --disable-keyboard-controls --topics $PLAY)
[ "$START_S" != 0 ] && ARGS+=(--start-offset "$START_S")
[ -n "$RELAY_IN" ] && ARGS+=(--remap "$RELAY_OUT:=$RELAY_IN")
setsid ros2 bag play /bag "${ARGS[@]}" </dev/null >"$L/play.log" 2>&1 & PLAY_PID=$!
STATUS_PLAY=playing
T0=$SECONDS; shown=0
# Humble's ros2 bag play has no --playback-duration: a slice ends by stopping
# the player once its length has played (at --rate, in wall time)
END=$(python3 -c "import sys; d, r = float(sys.argv[1]), float(sys.argv[2]); print(1 + d / r if d > 0 else 0)" \
      "$DURATION_S" "${RATE:-1.0}")
SLICED=0
while kill -0 "$PLAY_PID" 2>/dev/null; do
    if [ "$SLICED" = 0 ] && python3 -c "import sys; sys.exit(0 if 0 < float(sys.argv[1]) <= float(sys.argv[2]) else 1)" \
           "$END" "$((SECONDS - T0))"; then
        SLICED=1; kill -INT -- -"$PLAY_PID" 2>/dev/null
    fi
    if [ "$STATUS_METHOD" = running ] && ! kill -0 "$SLAM" 2>/dev/null; then
        STATUS_METHOD=exited_during_play
        echo "the method exited during the play:" >&2; tail -n 30 "$L/method.log" >&2
    fi
    if [ $((SECONDS - T0)) -ge $((shown + 30)) ]; then
        shown=$((SECONDS - T0)); echo "    ${shown} s"
    fi
    sleep 1
done
wait "$PLAY_PID"; code=$?; PLAY_PID=""
[ "$SLICED" = 1 ] && code=0
STATUS_PLAY=$([ "$code" = 0 ] && echo done || echo "failed ($code)")
[ "$code" = 0 ] || tail -n 20 "$L/play.log" >&2
echo "bag done after $((SECONDS - T0)) s; ${DRAIN:-5} s for the last outputs"
sleep "${DRAIN:-5}"
[ "$STATUS_METHOD" = running ] && STATUS_METHOD=ran
finish
[ -s "$W/trajectory.tum" ] || { echo "no trajectory written (logs: $L)" >&2; exit 3; }
