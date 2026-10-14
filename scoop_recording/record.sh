#!/bin/bash
# Start the bag for one machine and one pass, from record.yaml.
#
#   ./record.sh mobile_1 survey_1
#   ./record.sh mobile_1/sniffer coop
#   ./record.sh mobile_1 survey_1 slowwalk     # free-form label in the name
#
# Which topics: the profile named by the start of the pass (map_1 -> map, ...)
# picks groups per machine; PROFILE=, ONLY=a,b and SKIP=a,b override it. See
# the profiles: block in record.yaml. A machine a profile lists as [] does not
# record.
#
# Refuses to start if a listed topic has no publisher: a bag that silently
# contains an empty topic is worse than no bag, because you only find out after
# the robot has been driven. When you stop it (Ctrl-C) it checks the finished
# bag and exits non-zero if a topic has no messages (topics matching
# may_be_empty: in record.yaml only warn). It also writes ./.last_run for svo.sh.
set -eo pipefail

MACHINE=${1:?usage: record.sh <machine> <pass> [label]}
PASS=${2:?usage: record.sh <machine> <pass> [label]}
LABEL=${3:-${LABEL:-}}
HERE=$(cd "$(dirname "$0")" && pwd)
YAML=${RECORD_YAML:-$HERE/record.yaml}
QOS=${QOS_OVERRIDES:-$HERE/rosbag_qos_override.yaml}

[ -f "$YAML" ] || { echo "no $YAML"; exit 1; }

# ---------------------------------------------------------------- plan
PLAN=$(python3 - "$YAML" "$MACHINE" "$PASS" "$LABEL" <<'PYEOF'
import os, sys, datetime, yaml

path, machine, pas, label = (sys.argv[1], sys.argv[2],
                             sys.argv[3], sys.argv[4])
doc = yaml.safe_load(open(path)) or {}
d   = doc.get("defaults") or {}
machines = doc.get("machines") or {}

if machine not in machines:
    sys.exit("record.yaml has no machine %r (have: %s)"
             % (machine, ", ".join(sorted(machines))))

m = machines[machine]
topics = list(m.get("topics") or [])          # always recorded

# Optional topic groups: record.yaml `groups:` per machine, picked per run.
#   1. ONLY=a,b          exactly these (env)
#   2. PROFILE=name      profiles.<name>.<machine> = [groups]  (env)
#   3. profiles.<key>    key is a prefix of the pass name (survey_1 -> survey)
#   4. otherwise         all groups
# SKIP=a,b then removes groups from whatever was chosen.
groups = m.get("groups") or {}
profiles = doc.get("profiles") or {}
profile = os.environ.get("PROFILE") or next(
    (k for k in profiles if pas.startswith(str(k))), "")
if os.environ.get("PROFILE") and profile not in profiles:
    sys.exit("no profile %r in record.yaml (have: %s)"
             % (profile, ", ".join(sorted(map(str, profiles))) or "none"))
chosen = list(groups)
if os.environ.get("ONLY"):
    chosen = [g for g in os.environ["ONLY"].split(",") if g]
elif profile and machine in (profiles.get(profile) or {}):
    chosen = list(profiles[profile][machine] or [])
    if not chosen and not topics:
        # `machine: []` in a profile = this machine sits this run out.
        print("SKIP")
        print("profile %r: %s does not record this run" % (profile, machine))
        sys.exit(0)
chosen = [g for g in chosen
          if g not in (os.environ.get("SKIP") or "").split(",")]
bad = [g for g in chosen if g not in groups]
if bad:
    sys.exit("%s has no group(s) %s (have: %s)"
             % (machine, ", ".join(bad), ", ".join(groups) or "none"))
for g in chosen:
    topics += list(groups[g] or [])
summary = "profile=%s groups=%s" % (profile or "-", ",".join(chosen) or "-")

# Exactly one machine carries /tf and /tf_static. Two recorders give duplicated
# transforms at slightly different stamps, which breaks lookups on replay.
if doc.get("tf_recorded_by") == machine:
    topics += ["/tf", "/tf_static"]

# <prefix>_<pass>_[<site>_]<YYYYMMDD>_<machine>
# The machine goes last so that `ls mirc_dataset_survey_1_*` groups one run's
# four bags together, which is how you check a pass is complete.
stamp = datetime.date.today().strftime("%Y%m%d")
site = str(os.environ.get("SITE") or d.get("site") or "").strip()
parts = [d.get("prefix", "bag"), pas]
if site:
    parts.append(site)
if label:
    parts.append(label)
parts += [stamp, machine.replace("/", "_")]
out = "_".join(parts)

# out_dir may carry {label}, {pass}, {site}, {date} or {machine}. mkdir -p in
# the caller creates it, parents included, so a new session folder needs no
# setup -- but an unset placeholder would silently collapse the path, so that
# is an error rather than a surprise.
out_dir = machines[machine].get("out_dir") or d.get("out_dir") or "."
fields = {"label": label, "pass": pas, "site": site, "date": stamp,
          "machine": machine.replace("/", "_")}
for key, val in fields.items():
    token = "{%s}" % key
    if token in out_dir:
        if not val:
            sys.exit("out_dir is %r but %s is empty -- pass it: "
                     "record.sh <machine> <pass> <label>" % (out_dir, token))
        out_dir = out_dir.replace(token, str(val))

print(out)
print(os.path.expanduser(out_dir))
print(d.get("storage", "mcap"))
print(d.get("max_bag_size", 2147483648))
print(d.get("compression_mode", "none"))
print(" ".join(topics))
print(summary)
print(d.get("max_cache_size", 0))
print(d.get("storage_preset", ""))
print(d.get("max_bag_duration", 0))
PYEOF
) || { echo "$PLAN"; exit 1; }

if [ "$(sed -n 1p <<<"$PLAN")" = "SKIP" ]; then
    sed -n 2p <<<"$PLAN"
    exit 0
fi

OUT=$(sed -n 1p <<<"$PLAN")
OUT_DIR=$(sed -n 2p <<<"$PLAN")
STORAGE=$(sed -n 3p <<<"$PLAN")
MAXSIZE=$(sed -n 4p <<<"$PLAN")
COMPRESS=$(sed -n 5p <<<"$PLAN")
read -r -a TOPICS <<<"$(sed -n 6p <<<"$PLAN")"
SELECTED=$(sed -n 7p <<<"$PLAN")
CACHE=$(sed -n 8p <<<"$PLAN")
PRESET=$(sed -n 9p <<<"$PLAN")
DURATION=$(sed -n 10p <<<"$PLAN")

# Created if absent, parents included -- a fresh machine or a newly mounted
# drive needs no setup.
mkdir -p "$OUT_DIR" || { echo "cannot create $OUT_DIR"; exit 1; }
OUT="$OUT_DIR/${BAG_NAME:-$OUT}"

# A second run of the same pass on the same day would otherwise collide, and
# `ros2 bag record -o` refuses an existing directory -- after the robot has
# been driven to the start point. Take the next free _r<N> instead. This has
# to come after OUT_DIR is prepended, or it tests the wrong path.
if [ -e "$OUT" ]; then
    n=2
    while [ -e "${OUT}_r${n}" ]; do n=$((n + 1)); done
    echo "note: $(basename "$OUT") exists, using $(basename "${OUT}_r${n}")"
    OUT="${OUT}_r${n}"
fi

echo "machine  : $MACHINE"
echo "pass     : $PASS${LABEL:+  label: $LABEL}"
echo "output   : $OUT"
echo "free     : $(df -h "$OUT_DIR" | awk 'NR==2 {print $4}') on $(df -h "$OUT_DIR" | awk 'NR==2 {print $6}')"
echo "selected : $SELECTED"
echo "topics   : ${#TOPICS[@]}"

# ---------------------------------------------------------------- preflight
# `ros2 topic list` is the cheap check; a topic missing here has no publisher.
echo
echo "checking publishers..."
LIVE=$(ros2 topic list 2>/dev/null || true)
MISSING=()
for t in "${TOPICS[@]}"; do
    grep -qxF "$t" <<<"$LIVE" || MISSING+=("$t")
done

if [ ${#MISSING[@]} -gt 0 ]; then
    echo
    echo "NOT PUBLISHED (${#MISSING[@]}):"
    printf '  %s\n' "${MISSING[@]}"
    echo
    echo "Start the missing nodes, or set ALLOW_MISSING=1 to record anyway."
    [ "${ALLOW_MISSING:-0}" = "1" ] || exit 1
    for m in "${MISSING[@]}"; do
        for i in "${!TOPICS[@]}"; do
            [ "${TOPICS[$i]}" = "$m" ] && unset 'TOPICS[i]'
        done
    done
    TOPICS=("${TOPICS[@]}")
fi
echo "all present."

# ---------------------------------------------------------------- record
# `if` rather than `[ ... ] && ...`: under `set -e` a false test as the last
# statement of the script would abort it.
ARGS=(-s "$STORAGE" --max-bag-size "$MAXSIZE" -o "$OUT")
if [ "$COMPRESS" != "none" ]; then
    ARGS+=(--compression-mode file --compression-format "$COMPRESS")
fi
if [ -f "$QOS" ]; then
    ARGS+=(--qos-profile-overrides-path "$QOS")
fi
# The writer buffers messages in RAM and flushes to disk; when the disk is slower
# than the sensors the buffer overflows and messages are DROPPED. A bigger
# buffer rides out bursts, and the mcap `fastwrite` preset trades a little file
# size for write speed. Neither helps if the disk is slower than the sustained
# data rate -- the check below reports what was lost.
if [ "${CACHE:-0}" != "0" ]; then
    ARGS+=(--max-cache-size "$CACHE")
fi
# New bag file every N seconds as well as every max_bag_size: whichever comes first.
if [ "${DURATION:-0}" != "0" ]; then
    ARGS+=(-d "$DURATION")
fi
if [ -n "$PRESET" ] && [ "$STORAGE" = "mcap" ]; then
    ARGS+=(--storage-preset-profile "$PRESET")
fi

# Remember which bag this is, so svo.sh can name the ZED file after it.
echo "$OUT" > "$OUT_DIR/.last_run"

echo
echo "ros2 bag record ${ARGS[*]} ${TOPICS[*]}"
echo
# Not `exec`: when you stop it with Ctrl-C the check below still runs. `trap :`
# (a handler, not "ignore") keeps this script alive while the recorder, which
# gets the same Ctrl-C, shuts down and writes its metadata.
LOG=$(mktemp)
trap : INT
set +e
# `tee -i`: Ctrl-C goes to the whole foreground group. A plain tee would die with
# it, the recorder would then lose its output while shutting down, and its
# metadata.yaml would never be written.
ros2 bag record "${ARGS[@]}" "${TOPICS[@]}" 2>&1 | tee -i "$LOG"
set -e
trap - INT
# rosbag2 prints "Total lost: N" whenever its cache overflowed.
# (awk, not grep: grep exits 1 when nothing matches, which under `set -e` would
# end the script silently before the check below runs.)
LOST=$(awk '/Total lost:/ {s += $NF} END {print s + 0}' "$LOG")
rm -f "$LOG"

# ---------------------------------------------------------------- check
# A bag that stopped without data on a topic you asked for is found out here,
# not after the robot has been driven away. Topics matching `may_be_empty:` in
# record.yaml (event-driven ones like ntp/events, iperf outside a test) only
# warn.
echo
echo "== check: $(basename "$OUT") =="
CHECK=0
python3 - "$OUT" "$YAML" "${TOPICS[@]}" <<'PYEOF' || CHECK=$?
import fnmatch, os, sys, yaml

bag, yml, want = sys.argv[1], sys.argv[2], sys.argv[3:]
quiet = (yaml.safe_load(open(yml)) or {}).get("may_be_empty") or []
meta = os.path.join(bag, "metadata.yaml")
if not os.path.isfile(meta):
    sys.exit("FAIL  no metadata.yaml in %s -- the recorder did not finish writing" % bag)
info = (yaml.safe_load(open(meta)) or {})["rosbag2_bagfile_information"]
counts = {t["topic_metadata"]["name"]: t["message_count"]
          for t in info.get("topics_with_message_count") or []}
dur = info.get("duration", {}).get("nanoseconds", 0) / 1e9
size = sum(os.path.getsize(os.path.join(bag, f)) for f in os.listdir(bag)) / 1e6
print("duration %.1f s   size %.0f MB   messages %d" % (dur, size, sum(counts.values())))
empty = [t for t in want if counts.get(t, 0) == 0]
bad  = [t for t in empty if not any(fnmatch.fnmatch(t, q) for q in quiet)]
warn = [t for t in empty if t not in bad]
for t in warn:
    print("  quiet  %s" % t)
for t in bad:
    print("  EMPTY  %s" % t)
if bad:
    sys.exit("FAIL  %d of %d topics have no messages" % (len(bad), len(want)))
print("OK    %d topics have data%s" % (len(want) - len(empty),
      ", %d quiet (allowed)" % len(warn) if warn else ""))
PYEOF
if [ "$LOST" -gt 0 ]; then
    echo "FAIL  the recorder dropped $LOST messages (write cache overflowed)."
    echo "      Use a faster disk, lower the data rate (smaller images), or raise"
    echo "      max_cache_size in record.yaml. This pass is not complete."
    CHECK=1
fi
echo
exit $CHECK
