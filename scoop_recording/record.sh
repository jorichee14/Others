#!/bin/bash
# Start the bag for one machine and one pass, from record.yaml.
#
#   ./record.sh mobile_1 survey_1
#   ./record.sh mobile_1/sniffer coop
#   ./record.sh mobile_1 survey_1 slowwalk     # free-form label in the name
#
# Refuses to start if a listed topic has no publisher: a bag that silently
# contains an empty topic is worse than no bag, because you only find out after
# the robot has been driven.
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

topics = list(machines[machine].get("topics") or [])

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
PYEOF
) || { echo "$PLAN"; exit 1; }

OUT=$(sed -n 1p <<<"$PLAN")
OUT_DIR=$(sed -n 2p <<<"$PLAN")
STORAGE=$(sed -n 3p <<<"$PLAN")
MAXSIZE=$(sed -n 4p <<<"$PLAN")
COMPRESS=$(sed -n 5p <<<"$PLAN")
read -r -a TOPICS <<<"$(sed -n 6p <<<"$PLAN")"

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

echo
echo "ros2 bag record ${ARGS[*]} ${TOPICS[*]}"
echo
exec ros2 bag record "${ARGS[@]}" "${TOPICS[@]}"